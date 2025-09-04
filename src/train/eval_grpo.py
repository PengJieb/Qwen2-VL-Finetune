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
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
import ast
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration, HfArgumentParser, Qwen2_5_VLForConditionalGeneration
from transformers import AutoTokenizer
from src.model.qwenvl_more_modality import Qwen2_5_VLForConditionalGenerationMore, assign_qformer
from src.trainer.grpo_trainer import process_vision_info_more, extract_vision_info
from qwen_vl_utils import fetch_image
from src.dataset import make_supervised_data_module, make_supervised_eval_data_module, make_grpo_nextqa_data_module
from src.dataset.data_utils import replace_image_tokens
from src.params import DataArguments, ModelArguments, TrainingArguments
from src.utils import load_pretrained_model, get_model_name_from_path, disable_torch_init
from train.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_trainer
import pathlib
from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl, apply_liger_kernel_to_qwen2_5_vl
from monkey_patch_forward import replace_qwen2_5_with_mixed_modality_forward, replace_qwen_2_with_mixed_modality_forward
import argparse
local_rank = None
import json
from tqdm import tqdm
import re
from src.constants import (
    IGNORE_INDEX,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    SYSTEM_MESSAGE,
    GRPO_MESSAGE
)
import copy
import random

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
    return image_inputs, depth_inputs, norm_inputs, flow_inputs, None, text_inputs

def major_vote(lst):
    # Step 1: Count frequency of each value (preserve first occurrence order)
    frequency = {}
    order = []  # Track order of first occurrence to resolve ties
    for value in lst:
        if value not in frequency:
            frequency[value] = 1
            order.append(value)  # Record first time the value appears
        else:
            frequency[value] += 1

    # Step 2: Find the highest frequency
    max_freq = max(frequency.values())

    # Step 3: Collect all values with max frequency (in first-occurrence order)
    candidates = [val for val in order if frequency[val] == max_freq]

    # Step 4: Return the first candidate (resolves ties)
    return candidates[0]

def uncertainty(scores):
    entropy = []
    topk_uncertainty = []
    variance = []
    epsilon = 1e-10
    for ss in scores:
        probs = F.softmax(ss, dim=-1)
        entropy.append(-torch.sum(probs * torch.log(probs + epsilon), dim=-1))
        
        topk_probs, _ = torch.topk(probs, k=5, dim=-1)
        topk_sum = torch.sum(topk_probs, dim=-1)
        topk_uncertainty.append(1-topk_sum)
        
        indices = torch.arange(probs.shape[-1], device=probs.device).unsqueeze(0)
        mean = torch.sum(indices * probs, dim=-1)
        variance.append(torch.sum(probs * (indices - mean.unsqueeze(1))** 2, dim=-1))
    print(torch.mean(torch.stack(entropy)), torch.mean(torch.stack(topk_uncertainty)), torch.mean(torch.stack(variance)))
    return None

def eval():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    lora_enable = training_args.lora_enable
    
    processor, model = load_pretrained_model(model_base = model_args.model_id, model_path = model_args.model_path, 
                                                device_map=training_args.device, model_name=model_args.model_path, 
                                                load_4bit= training_args.bits==4,load_8bit=training_args.bits==8,
                                                device=training_args.device, use_flash_attn=not training_args.disable_flash_attn2,
                                                n_image=model_args.n_image, n_depth=model_args.n_depth,
                                                n_norm=model_args.n_norm, n_flow=model_args.n_flow, n_prefusion_layers=model_args.n_prefusion_layers,
                                                multilevel_qformer=model_args.multilevel_qformer,
                                                lora_enable = lora_enable, grpo_pretrain = model_args.sft_model_path
                        )

    data_module = make_grpo_nextqa_data_module(model_id=model_args.model_id,
                                              processor=processor,
                                              data_args=data_args)

    # print('Loading additional Qwen2-VL weights...')
    # non_lora_trainables = torch.load(os.path.join(model_args.model_path, 'non_lora_state_dict.bin'), map_location='cpu')
    # non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
    # if any(k.startswith('model.model.') for k in non_lora_trainables):
    #     non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
    # model.load_state_dict(non_lora_trainables, strict=False)
    n_times = 5
    dataloader = DataLoader(data_module['train_dataset'], batch_size=1, collate_fn=data_module['data_fn'], shuffle=False)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_id, trust_remote_code=True)
    generation_args = {
        "max_new_tokens": 512,
        "temperature": 0.95,
        "do_sample": True,
        "repetition_penalty": 1.0,
        "top_p": 0.95,
        "output_scores": True,
        "return_dict_in_generate": True
    }
    device = training_args.device
    model = model.to(device, dtype=torch.bfloat16)
    model.eval()
    group_acc = {'CW': 0, 'CH': 0, 'TN': 0, 'TC': 0, 'DC': 0, 'DL': 0, 'DO': 0, 'TP': 0}
    group_cnt = {'CW': 0, 'CH': 0, 'TN': 0, 'TC': 0, 'DC': 0, 'DL': 0, 'DO': 0, 'TP': 0}
    overall_acc = {'C':0, 'T':0, 'D':0}
    overall_cnt = {'C':0, 'T':0, 'D':0}
    all_acc = 0
    all_cnt = 0
    for batch in tqdm(dataloader):
        all_cnt += 1
        # batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
        qid = batch[0].pop('qid')
        # print(batch)
        prompts = [x["prompt"] for x in batch]
        labels=batch[0]['assistant']['content']
        system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
        inner_pred = []
        think_processes = []
        for j in range(n_times):
            
            image_inputs, depth_inputs, norm_inputs, flow_inputs, video_inputs, text_inputs = process_vision_info_more(prompts, return_video_kwargs=False, is_random=False)
            pre_info = '\n'.join(think_processes)

            prompts_text = [
                system_message + f"{DEFAULT_IM_START_TOKEN}{'user'}\n{replace_image_tokens(item)}{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}assistant\n"
                for item in text_inputs
            ]

            # print(prompts_text)
            
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
            # print(tokenizer.decode(batch['input_ids'][0], skip_special_tokens=False))
            out = model.generate(eos_token_id=processor.tokenizer.eos_token_id, **prompt_inputs, **generation_args)
            # print(out.keys())
            # for kk in out:
            #     print(out[kk].shape)
            # uncertainty(out['scores'])
            # print(out['scores'][0].shape, out['scores'][0].softmax(dim=-1), len(out['scores']), out['sequences'].shape)
            pred_sequence = out['sequences']
            pred = tokenizer.decode(pred_sequence[0], skip_special_tokens=False)
            # print(pred)
            pred = pred.split('assistant')[-1][1:]
            # label_index = batch['labels'][0]
            # label_index = label_index[label_index!= -100]
            # label = tokenizer.decode(label_index, skip_special_tokens=True)
            
            qtype = qid.split('_')[0]
            # print(labels)
            # print(pred)
            # print(labels)
            # a_label = label[0]
            think_process = re.search(r"<think>(.*?)</think>", pred, re.S)
            if think_process is not None:
                think_processes.append(think_process.group(1).strip())
                print(think_processes[-1])
            
            label_match = re.search(r"<answer>(.*?)</answer>", labels, re.S)
            ground_truth = label_match.group(1).strip() if label_match is not None else labels.strip()
            if "</think>" in pred:
                pred = pred.split('</think>')[1]
            pred_match = re.search(r"<answer>(.*?)</answer>", pred, re.S)
            if pred_match is None:
                # group_acc[qtype] += 1
                # overall_acc[qtype[0]] += 1
                # all_acc += 1
                continue
            student_answer = pred_match.group(1).strip() if pred_match is not None else pred_match.strip()
            # print(ground_truth)
            # print(student_answer)
            if len(student_answer) == 0:
                # group_acc[qtype] += 1
                # overall_acc[qtype[0]] += 1
                # all_acc += 1
                continue
            # a_pred = pred[0]
            # print(a_label, a_pred)
            # if all_acc
            
            # group_cnt[qtype] += 1
            # overall_cnt[qtype[0]] += 1
            # inner_pred.append(student_answer)
            i_pred = student_answer[:2]
            if i_pred[0] in ['0', '1', '2', '3', '4', '5', '6']:
                inner_pred.append(i_pred[0])
            else:
                inner_pred.append(i_pred[1])
        print(inner_pred)
        print(ground_truth)
        student_answer = major_vote(inner_pred)
        # student_answer = inner_pred[0]
        if len(inner_pred) < 1:
            all_cnt += 1
        else:
            if student_answer[0] == ground_truth[0]:
                group_acc[qtype] += 1
                overall_acc[qtype[0]] += 1
                all_acc += 1
            else:
                if ':' in student_answer and ':' in ground_truth:
                    if student_answer.split(':')[0][-2:]==ground_truth.split(':')[0][-2:]:
                        group_acc[qtype] += 1
                        overall_acc[qtype[0]] += 1
                        all_acc += 1
        group_cnt[qtype] += 1
        overall_cnt[qtype[0]] += 1
        # inner_pred.append(student_answer)
        print('Acc: {:.2f}'.format(all_acc*100.0/(all_cnt+0.00001)))
        # print(label, pred)
        # print(a_label, a_pred)
        # print(len(a_label), len(a_pred))
        # break
    results = {}
    print('Acc: {:.2f}'.format(all_acc*100.0/(all_cnt+0.00001)))
    results['ave'] = all_acc*100.0/all_cnt
    for qtype in group_acc:
        results[qtype] = group_acc[qtype]*100.0/(group_cnt[qtype]+0.00001)
    for sqtype in overall_acc:
        results[sqtype] = overall_acc[sqtype]*100.0/(overall_cnt[sqtype]+0.00001)
    print(results)
    with open(os.path.join(model_args.model_path, 'eval_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    # trainer = QwenSFTTrainer(
    #     model=model,
    #     processing_class=processor,
    #     args=training_args,
    #     **data_module
    # )
    
    # if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
    #     trainer.train(resume_from_checkpoint=True)
    # else:
    #     trainer.train()

    # trainer.save_state()

    # model.config.use_cache = True
    
    # if training_args.lora_enable:
    #     state_dict = get_peft_state_maybe_zero_3(
    #         model.named_parameters(), training_args.lora_bias
    #     )

    #     non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
    #         model.named_parameters(), require_grad_only=True
    #     )

    #     if local_rank == 0 or local_rank == -1:
    #         model.config.save_pretrained(training_args.output_dir)
    #         model.save_pretrained(training_args.output_dir, state_dict=state_dict)
    #         torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_state_dict.bin"))
    # else:
    #     safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)



if __name__ == "__main__":
    eval()