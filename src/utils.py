from peft import PeftModel
import torch
from transformers import BitsAndBytesConfig, Qwen2VLForConditionalGeneration, AutoProcessor, AutoConfig, Qwen2_5_VLForConditionalGeneration
from src.model.qwenvl_more_modality import Qwen2_5_VLForConditionalGenerationMore, assign_qformer, assign_prefusion, Qwen2_5_VLProcessorOneToken
import warnings
import os
import json
import importlib
import inspect
from types import ModuleType
from typing import Callable, List

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

# This code is borrowed from LLaVA
def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, 
                          device_map="auto", device="cuda", use_flash_attn=False, grpo_pretrain = None, **gkwargs):
    kwargs = {"device_map": device_map}
    
    if device != "cuda":
        kwargs['device_map'] = {"":device}
    
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    
    lora_enable = gkwargs.get('lora_enable', False)
    
    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    if lora_enable and model_base is None:
        warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument.')
    if lora_enable and model_base is not None and grpo_pretrain is None:
        lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
        if hasattr(lora_cfg_pretrained, 'quantization_config'):
            del lora_cfg_pretrained.quantization_config
        # processor = AutoProcessor.from_pretrained(model_base)
        processor = Qwen2_5_VLProcessorOneToken.from_pretrained(model_base, n_frames = gkwargs['n_image']+gkwargs['n_depth']+gkwargs['n_norm']+gkwargs['n_flow'])
        print('Loading Qwen2-VL from base model...')
        if "Qwen2.5" in model_base:
            print(model_base)
            model = Qwen2_5_VLForConditionalGenerationMore.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            assign_qformer(model, {"image": gkwargs['n_image'], 'depth': gkwargs['n_depth'], 'norm': gkwargs['n_norm'], 'flow': gkwargs['n_flow']},
                           multilevel_qformer=gkwargs['multilevel_qformer'])
            assign_prefusion(model, gkwargs['n_prefusion_layers'])
        else:
            model = Qwen2VLForConditionalGeneration.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
        token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

        print('Loading additional Qwen2-VL weights...')
        non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_state_dict.bin'), map_location='cpu')
        print(non_lora_trainables.keys())
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        print(non_lora_trainables.keys())
        model.load_state_dict(non_lora_trainables, strict=False)
    
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, model_path)

        print('Merging LoRA weights...')
        model = model.merge_and_unload()

        print('Model Loaded!!!')
    elif grpo_pretrain is not None:
        
        lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
        if hasattr(lora_cfg_pretrained, 'quantization_config'):
            del lora_cfg_pretrained.quantization_config
        # processor = AutoProcessor.from_pretrained(model_base)
        processor = Qwen2_5_VLProcessorOneToken.from_pretrained(model_base, n_frames = gkwargs['n_image']+gkwargs['n_depth']+gkwargs['n_norm']+gkwargs['n_flow'])
        print('Loading Qwen2-VL from base model...')
        if "Qwen2.5" in model_base:
            print(model_base)
            model = Qwen2_5_VLForConditionalGenerationMore.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            assign_qformer(model, {"image": gkwargs['n_image'], 'depth': gkwargs['n_depth'], 'norm': gkwargs['n_norm'], 'flow': gkwargs['n_flow']},
                           multilevel_qformer=gkwargs['multilevel_qformer'])
            assign_prefusion(model, gkwargs['n_prefusion_layers'])
            model.from_pretrained(grpo_pretrain)
        else:
            model = Qwen2VLForConditionalGeneration.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
        token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
        if lora_enable:
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)

            print('Merging LoRA weights...')
            model = model.merge_and_unload()

        print('Model Loaded!!!')
    else:
        with open(os.path.join(model_path, 'config.json'), 'r') as f:
            config = json.load(f)

        if "Qwen2_5" in config["architectures"][0]:
            # processor = AutoProcessor.from_pretrained(model_path)
            processor = Qwen2_5_VLProcessorOneToken.from_pretrained(model_base, n_frames = gkwargs['n_image']+gkwargs['n_depth']+gkwargs['n_norm']+gkwargs['n_flow'])
            # print(model_base)
            model_ref = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            kwargs['n_image'] = gkwargs['n_image']
            kwargs['n_depth'] = gkwargs['n_depth']
            kwargs['n_norm'] = gkwargs['n_norm']
            kwargs['n_flow'] = gkwargs['n_flow']
            kwargs['multilevel_qformer'] = gkwargs['multilevel_qformer']
            kwargs['n_prefusion_layers'] = gkwargs['n_prefusion_layers']
            model = Qwen2_5_VLForConditionalGenerationMore.from_pretrained(model_path, low_cpu_mem_usage=True, eval_model = True,
                                                                           **kwargs)
            model_dict = {pn:p for pn, p in model.named_parameters()}
            with torch.no_grad():
                for pn, p in model_ref.named_parameters():
                    model_dict[pn].copy_(model_dict[pn])
            del(model_ref)
        else:
            processor = AutoProcessor.from_pretrained(model_path)
            model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    return processor, model


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]
    
def load_reward_funcs(
    module_path: str = "train.reward_funcs",
    *,
    name_pred = lambda n: n.endswith("_reward"),
    obj_pred  = lambda o: callable(o),
    keep_order: bool = True
) -> List[Callable]:

    mod: ModuleType = importlib.import_module(module_path)
    
    members = inspect.getmembers(mod, predicate=obj_pred)

    reward_funcs = [(n, o) for n, o in members if name_pred(n)]

    if keep_order:
        reward_funcs.sort(key=lambda pair: inspect.getsourcelines(pair[1])[1])

    return [o for _, o in reward_funcs]