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
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from peft import LoraConfig, get_peft_model
import ast
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration, HfArgumentParser, Qwen2_5_VLForConditionalGeneration
from transformers import AutoTokenizer
from src.model.qwenvl_more_modality import Qwen2_5_VLForConditionalGenerationMore, assign_qformer
from src.trainer import QwenSFTTrainer
from src.trainer.grpo_trainer import extract_vision_info
from qwen_vl_utils import fetch_image
from src.dataset import make_supervised_data_module, make_supervised_eval_data_module, make_supervised_eval_data_module2
from src.params import DataArguments, ModelArguments, TrainingArguments
from src.utils import load_pretrained_model, get_model_name_from_path, disable_torch_init
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
import deepspeed

from src.constants import (
    IGNORE_INDEX,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    SYSTEM_MESSAGE,
    GRPO_MESSAGE
)

local_rank = int(os.environ.get("LOCAL_RANK", -1))
torch.cuda.set_device(local_rank) if local_rank != -1 else None

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
    n_frame = 10, is_random = True
):

    vision_infos = extract_vision_info(conversations)
    # print(vision_infos)
    ## Read images or videos
    image_inputs = []
    depth_inputs = []
    norm_inputs = []
    flow_inputs = []
    text_inputs = []
    
    image_info = []
    depth_info = []
    norm_info = []
    flow_info = []
    
    for vision_info in vision_infos:
        if 'type' in vision_info:
            if 'image' == vision_info['type']:
                image_info.append(vision_info)
            elif 'depth' == vision_info['type']:
                depth_info.append(vision_info)
            elif 'norm' == vision_info['type']:
                norm_info.append(vision_info)
            elif 'flow' == vision_info['type']:
                flow_info.append(vision_info)
            elif 'text' == vision_info['type']:
                text_inputs.append(vision_info['text'])
            else:
                raise ValueError("image, image_url or video should in content.")
    
    n_frames = min(len(image_info), len(depth_info), len(norm_info), len(flow_info))
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
        flow_inputs.append(fetch_image(flow_info[sid]))
    if len(image_inputs) == 0:
        image_inputs = None

    # print(len(image_inputs), len(depth_inputs), len(norm_inputs), len(flow_inputs))
    return image_inputs, depth_inputs, norm_inputs, flow_inputs, text_inputs

def eval():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    ds_config_path = training_args.deepspeed
    lora_enable = training_args.lora_enable
    rank0_print(f"Loading model (DeepSpeed will handle device placement)")
    processor, model = load_pretrained_model(model_base = model_args.model_id, model_path = model_args.model_path, 
                                                device_map="auto", model_name=model_args.model_path, 
                                                load_4bit= training_args.bits==4,load_8bit=training_args.bits==8,
                                                device=training_args.device, use_flash_attn=not training_args.disable_flash_attn2,
                                                n_image=model_args.n_image, n_depth=model_args.n_depth,
                                                n_norm=model_args.n_norm, n_flow=model_args.n_flow, n_prefusion_layers=model_args.n_prefusion_layers,
                                                multilevel_qformer=model_args.multilevel_qformer, multilevel_mlp=model_args.multilevel_mlp,
                                                lora_enable = lora_enable
                        )

    model, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters() if lora_enable else None,  # Pass params only if LoRA is enabled
        config_params=json.load(open(ds_config_path)),  # Load DeepSpeed config
        dist_init_required=local_rank != -1,  # Auto - initialize distributed process group
    )
    model_module = model.module if hasattr(model, 'module') else model
    
    data_module = make_supervised_eval_data_module2(model_id=model_args.model_id,
                                              processor=processor,
                                              data_args=data_args)
    eval_dataset = data_module['train_dataset']
    rank0_print(f"Evaluation dataset size: {len(eval_dataset)}")
    sampler = DistributedSampler(eval_dataset, shuffle=False) if local_rank != -1 else None

    n_times = 1
    
    # dataloader = DataLoader(data_module['train_dataset'], batch_size=1, collate_fn=data_module['data_collator'], num_workers=8, shuffle=False)
    
    dataloader = DataLoader(
        eval_dataset,
        batch_size=1,  # Batch size per GPU (DeepSpeed handles gradient accumulation if needed)
        collate_fn=data_module['data_collator'],
        num_workers=8,
        shuffle=(sampler is None),
        sampler=sampler,
        pin_memory=True  # Speed up data transfer to GPU
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_id, trust_remote_code=True)
    generation_args = {
        "max_new_tokens": 2,
        "temperature": 0.95,
        "do_sample": True,
        "repetition_penalty": 1.0,
        "top_p": 0.95
    }
    device = training_args.device
    model = model.to(device, dtype=torch.bfloat16)
    model.eval()
    with torch.no_grad():
        group_acc = {'CW': 0, 'CH': 0, 'TN': 0, 'TC': 0, 'DC': 0, 'DL': 0, 'DO': 0, 'TP': 0}
        group_cnt = {'CW': 0, 'CH': 0, 'TN': 0, 'TC': 0, 'DC': 0, 'DL': 0, 'DO': 0, 'TP': 0}
        overall_acc = {'C':0, 'T':0, 'D':0}
        overall_cnt = {'C':0, 'T':0, 'D':0}
        all_acc = 0
        all_cnt = 0
        for batch in tqdm(dataloader, desc=f"Rank {local_rank} Evaluating"):
            all_cnt += 1
            # batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
            qid = batch[0].pop('qid')
        
            prompts = [x["prompt"] for x in batch]
            labels=batch[0]['assistant']['content']

            system_message=f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE} always describe image first“ + ”{DEFAULT_IM_END_TOKEN}\n"
        
            inner_pred = []
        
            image_inputs, depth_inputs, norm_inputs, flow_inputs, text_inputs = process_vision_info_more(prompts, return_video_kwargs=False, is_random=False)
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
                images=flow_inputs,
                videos=video_inputs,
                padding=False,
                do_resize=False,
                return_tensors="pt"
            )
            prompt_inputs['flow_values'] = tmp_prompt_inputs['pixel_values']
            
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
            
            prompt_inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in prompt_inputs.items()}
            
            for _ in range(n_times):
                # Use the DeepSpeed - wrapped model for generation
                out = model.generate(**prompt_inputs, **generation_args)
                pred = tokenizer.decode(out[0], skip_special_tokens=False)
                pred = pred.split('assistant')[-1].lstrip()
                inner_pred.append(pred[0] if len(pred) > 0 else "")
            # continue
            # out = model.generate(eos_token_id=processor.tokenizer.eos_token_id, **prompt_inputs, **generation_args)
            # pred = tokenizer.decode(out[0], skip_special_tokens=False)
            # # print(pred)
            # pred = pred.split('assistant')[-1][1:]
            # print(pred)
            # # label_index = batch['labels'][0]
            # # label_index = label_index[label_index!= -100]
            # label = labels.strip()
            # print(label)
            
            a_label = labels[0] if len(labels) > 0 else ""
            a_pred = inner_pred[0]
            qtype = qid.split('_')[0] if '_' in qid else ""

            group_cnt[qtype] += 1
            if a_label == a_pred:
                group_acc[qtype] += 1
                all_acc += 1

            s_type = qtype[0]
            if s_type in overall_cnt:
                overall_cnt[s_type] += 1
                if a_label == a_pred:
                    overall_acc[s_type] += 1

            if local_rank in (-1, 0):
                rank0_print(f"QID: {qid} | Label: {a_label} | Pred: {a_pred} | Current Acc: {all_acc*100/(all_cnt+1e-5):.2f}%")

        # print(label, pred)
        # print(a_label, a_pred)
        # print(len(a_label), len(a_pred))
        # break
        
    qtypes = ['CW', 'CH', 'TN', 'TC', 'DC', 'DL', 'DO', 'TP']
    overall_types = ['C', 'T', 'D']
    device = torch.device(f"cuda:{local_rank}") if local_rank != -1 else torch.device("cpu")

    # Convert local metrics to tensors
    local_group_acc = torch.tensor([group_acc[q] for q in qtypes], device=device, dtype=torch.float32)
    local_group_cnt = torch.tensor([group_cnt[q] for q in qtypes], device=device, dtype=torch.float32)
    local_overall_acc = torch.tensor([overall_acc[t] for t in overall_types], device=device, dtype=torch.float32)
    local_overall_cnt = torch.tensor([overall_cnt[t] for t in overall_types], device=device, dtype=torch.float32)
    local_all_acc = torch.tensor([all_acc], device=device, dtype=torch.float32)
    local_all_cnt = torch.tensor([all_cnt], device=device, dtype=torch.float32)
    
    if local_rank != -1:
        dist.all_reduce(local_group_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_group_cnt, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_overall_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_overall_cnt, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_all_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_all_cnt, op=dist.ReduceOp.SUM)
        
    if local_rank in (-1, 0):
        agg_group_acc = {qtypes[i]: local_group_acc[i].item() for i in range(len(qtypes))}
        agg_group_cnt = {qtypes[i]: local_group_cnt[i].item() for i in range(len(qtypes))}
        agg_overall_acc = {overall_types[i]: local_overall_acc[i].item() for i in range(len(overall_types))}
        agg_overall_cnt = {overall_types[i]: local_overall_cnt[i].item() for i in range(len(overall_types))}
        agg_all_acc = local_all_acc[0].item()
        agg_all_cnt = local_all_cnt[0].item()

        # Calculate final accuracies
        results = {
            "overall_accuracy": agg_all_acc * 100.0 / agg_all_cnt if agg_all_cnt > 0 else 0.0,
            "per_group_accuracy": {q: agg_group_acc[q] * 100.0 / agg_group_cnt[q] if agg_group_cnt[q] > 0 else 0.0 for q in qtypes},
            "per_category_accuracy": {t: agg_overall_acc[t] * 100.0 / agg_overall_cnt[t] if agg_overall_cnt[t] > 0 else 0.0 for t in overall_types}
        }

        # Log and save
        rank0_print("\n" + "="*50)
        rank0_print(f"Final Evaluation Results (Total Samples: {int(agg_all_cnt)})")
        rank0_print(f"Overall Accuracy: {results['overall_accuracy']:.2f}%")
        rank0_print("Per-Group Accuracy:")
        for q, acc in results['per_group_accuracy'].items():
            rank0_print(f"  {q}: {acc:.2f}%")
        rank0_print("Per-Category Accuracy:")
        for t, acc in results['per_category_accuracy'].items():
            rank0_print(f"  {t}: {acc:.2f}%")
        rank0_print("="*50)

        # Save results to JSON
        save_path = os.path.join(model_args.model_path, 'eval_results_deepspeed.json')
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
        rank0_print(f"Results saved to: {save_path}")



if __name__ == "__main__":
    eval()