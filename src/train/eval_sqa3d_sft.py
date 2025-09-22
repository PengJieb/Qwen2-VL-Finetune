'''
Author: PengJie pengjieb@mail.ustc.edu.cn
Date: 2025-06-12 22:39:55
LastEditors: PengJie pengjieb@mail.ustc.edu.cn
LastEditTime: 2025-06-26 20:31:04
FilePath: /Qwen2-VL-Finetune/src/train/eval_sft.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import torch
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
import ast
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration, HfArgumentParser, Qwen2_5_VLForConditionalGeneration
from transformers import AutoTokenizer
from src.model.qwenvl_more_modality import Qwen2_5_VLForConditionalGenerationMore, assign_qformer
from src.trainer import QwenSFTTrainer
from src.trainer.grpo_trainer import extract_vision_info
from qwen_vl_utils import fetch_image
from src.dataset import make_supervised_data_module, make_supervised_eval_data_module, make_supervised_eval_sqa3d_data_module
from src.params import DataArguments, ModelArguments, TrainingArguments
from src.utils import load_pretrained_model, get_model_name_from_path, disable_torch_init, load_pretrained_sqa3d_model
from src.dataset.data_utils import replace_image_tokens
from train.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_trainer
import pathlib
from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl, apply_liger_kernel_to_qwen2_5_vl
from monkey_patch_forward import replace_qwen2_5_with_mixed_modality_forward, replace_qwen_2_with_mixed_modality_forward
import argparse
local_rank = None
import json
from tqdm import tqdm
import random
import copy

from src.constants import (
    IGNORE_INDEX,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    SYSTEM_MESSAGE,
    GRPO_MESSAGE
)

def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)

def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)
    
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        rank0_print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names

def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad

def configure_vision_tower(model, training_args, compute_dtype, device):
    vision_tower = model.visual
    vision_tower.to(dtype=compute_dtype, device=device)

    vision_model_params = model.visual.parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)
    
    # Handle merger specifically
    merger_params = model.visual.merger.parameters()
    set_requires_grad(merger_params, not training_args.freeze_merger)

def configure_llm(model, training_args):
    lm_head = model.lm_head.parameters()
    set_requires_grad(lm_head, not training_args.freeze_llm)

    llm_params = model.model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)


def process_vision_info_more(
    conversations: list[dict] | list[list[dict]],
    return_video_kwargs: bool = False,
    n_frame = 5, is_random = True
):

    vision_infos = extract_vision_info(conversations)
    # print(vision_infos)
    ## Read images or videos
    image_inputs = []
    depth_inputs = []
    norm_inputs = []
    text_inputs = []
    pc_inputs = []
    pc_feature_inputs = []
    
    image_info = []
    depth_info = []
    norm_info = []
    pc_info = []
    
    for vision_info in vision_infos:
        if 'type' in vision_info:
            # print(vision_info['type'])
            if 'image' == vision_info['type']:
                image_info.append(vision_info)
            elif 'depth' == vision_info['type']:
                depth_info.append(vision_info)
            elif 'norm' == vision_info['type']:
                norm_info.append(vision_info)
            elif 'text' == vision_info['type']:
                text_inputs.append(vision_info['text'])
            elif 'pc' == vision_info['type']:
                pc_info.append(
                    {'pc': vision_info['pc'],
                    'pc_feature': vision_info['pc_feature']}
                )
            else:
                raise ValueError("image, image_url or video should in content.")
    
    n_frames = min(len(image_info), len(depth_info), len(norm_info))
    # print(f"N FRAME: {n_frames}")
    indices = [i for i in range(n_frames)]
    if is_random:
        selected_frame_ids = random.sample(indices, n_frame)
        selected_frame_ids.sort()
        
    else:
        indices = [int(i * (n_frames - 1) / (n_frame - 1)) for i in range(n_frame)]
        selected_frame_ids = sorted(list(set(indices)))
    # print(selected_frame_ids)
    for sid in selected_frame_ids:
        image_inputs.append(fetch_image(image_info[sid]))
        depth_inputs.append(fetch_image(depth_info[sid]))
        norm_inputs.append(fetch_image(norm_info[sid]))
    pc_inputs.append(pc_info[0]['pc'])
    pc_feature_inputs.append(pc_info[0]['pc_feature'])
    if len(image_inputs) == 0:
        image_inputs = None

    # print(len(image_inputs), len(depth_inputs), len(norm_inputs), len(flow_inputs))
    return image_inputs, depth_inputs, norm_inputs, pc_inputs, pc_feature_inputs, text_inputs

def eval():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    lora_enable = training_args.lora_enable
    
    processor, model = load_pretrained_sqa3d_model(model_base = model_args.model_id, model_path = model_args.model_path, 
                                                device_map=training_args.device, model_name=model_args.model_path, 
                                                load_4bit= training_args.bits==4,load_8bit=training_args.bits==8,
                                                device=training_args.device, use_flash_attn=not training_args.disable_flash_attn2,
                                                n_image=model_args.n_image, n_depth=model_args.n_depth,
                                                n_norm=model_args.n_norm, n_flow=model_args.n_flow, n_prefusion_layers=model_args.n_prefusion_layers,
                                                multilevel_qformer=model_args.multilevel_qformer, multilevel_mlp=model_args.multilevel_mlp, model_args = model_args,
                                                lora_enable = lora_enable
                        )

    data_module = make_supervised_eval_sqa3d_data_module(model_id=model_args.model_id,
                                              processor=processor,
                                              data_args=data_args)

    # print('Loading additional Qwen2-VL weights...')
    # non_lora_trainables = torch.load(os.path.join(model_args.model_path, 'non_lora_state_dict.bin'), map_location='cpu')
    # non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
    # if any(k.startswith('model.model.') for k in non_lora_trainables):
    #     non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
    # model.load_state_dict(non_lora_trainables, strict=False)

    n_times = 1
    
    dataloader = DataLoader(data_module['train_dataset'], batch_size=1, collate_fn=data_module['data_collator'], num_workers=8, shuffle=False)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_id, trust_remote_code=True)
    generation_args = {
        "max_new_tokens": 4,
        "temperature": 0.95,
        "do_sample": False,
        "repetition_penalty": 1.0,
        "top_p": 0.95
    }
    device = training_args.device
    model = model.to(device, dtype=torch.bfloat16)
    
    group_acc = {'What': 0, 'Is': 0, 'How': 0, 'Can': 0, 'Which': 0, 'Other': 0}
    group_cnt = {'What': 0, 'Is': 0, 'How': 0, 'Can': 0, 'Which': 0, 'Other': 0}
    all_acc = 0
    all_cnt = 0
    for batch in tqdm(dataloader):
        all_cnt += 1
        # batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
        qid = batch[0].pop('qid')
        
        prompts = [x["prompt"] for x in batch]
        labels=batch[0]['assistant']['content']
        # print(labels)
        system_message=f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE} always describe image first“ + ”{DEFAULT_IM_END_TOKEN}\n"
        
        inner_pred = []
        for j in range(n_times):
        
            image_inputs, depth_inputs, norm_inputs, pc_inputs, pc_feature_inputs, text_inputs = process_vision_info_more(prompts, return_video_kwargs=False, is_random=False)
            video_inputs = None
            prompts_text = [
                system_message + f"{DEFAULT_IM_START_TOKEN}{'user'}\n{replace_image_tokens(item)}{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}assistant\n"
                for item in text_inputs
            ]

            tmp_prompt_text = copy.deepcopy(prompts_text)
            prompt_inputs = processor(
                text = prompts_text,
                images=image_inputs,
                videos=video_inputs,
                padding=False,
                do_resize=False,
                return_tensors="pt"
            )
            tmp_prompt_inputs = processor(
                text = copy.deepcopy(tmp_prompt_text),
                images=depth_inputs,
                videos=video_inputs,
                padding=False,
                do_resize=False,
                return_tensors="pt"
            )
            prompt_inputs['depth_values'] = tmp_prompt_inputs['pixel_values']
            
            tmp_prompt_inputs = processor(
                text = copy.deepcopy(tmp_prompt_text),
                images=norm_inputs,
                videos=video_inputs,
                padding=False,
                do_resize=False,
                return_tensors="pt"
            )
            prompt_inputs['norm_values'] = tmp_prompt_inputs['pixel_values']
            prompt_inputs['norm_value_grid'] = tmp_prompt_inputs['image_grid_thw']
            prompt_inputs['pc_values'] = torch.stack(pc_inputs).to(torch.bfloat16).to(device)
            prompt_inputs['pc_feature_values'] = torch.stack(pc_feature_inputs).to(torch.bfloat16).to(device)
            
            prompt_inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in prompt_inputs.items()}

            # continue
            out = model.generate(eos_token_id=processor.tokenizer.eos_token_id, **prompt_inputs, **generation_args)
            pred = tokenizer.decode(out[0], skip_special_tokens=False)
            # print(pred)
            pred = pred.split('assistant')[-1][1:]
            pred = pred.split(' ')
            print(pred, labels)
            # label_index = batch['labels'][0]
            # label_index = label_index[label_index!= -100]
            label = labels.split('<')[0].strip().lower()
            # print(label)
            
            qtype = qid.split('_')[0]
            
            a_label = label
            a_pred = pred[0].lower()
            inner_pred.append(a_pred)
        # print(a_label, inner_pred)
        # if all_acc
        a_pred = inner_pred[0]
        if a_label == a_pred:
            group_acc[qtype] += 1
            all_acc += 1
        group_cnt[qtype] += 1

        print('Acc: {:.2f}'.format(all_acc*100.0/(all_cnt+0.00001)))


    results = {}
    print('Acc: {:.2f}'.format(all_acc*100.0/(all_cnt+0.00001)))
    results['ave'] = all_acc*100.0/all_cnt
    for qtype in group_acc:
        results[qtype] = group_acc[qtype]*100.0/(group_cnt[qtype]+0.00001)
    print(results)
    with open(os.path.join(model_args.model_path, 'eval_results.json'), 'w') as f:
        json.dump(results, f, indent=4)



if __name__ == "__main__":
    eval()