#!/bin/bash

# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
export HOME=/playpen/pengjie_xinyu
export PYTHONPATH=src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1
python src/merge_lora_weights.py \
    --model-path output_local/lora_vision_test_32_32_32_32_token_dynamic_full_crema_policy_fullfinetune_3b_pretrain-336-1 \
    --model-base $MODEL_NAME  \
    --save-model-path output_local/grpo_pretrain \
    --safe-serialization