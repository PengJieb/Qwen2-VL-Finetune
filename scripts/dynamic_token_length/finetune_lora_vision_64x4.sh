#!/bin/bash
###
 # @Author: PengJie pengjieb@mail.ustc.edu.cn
 # @Date: 2025-06-12 19:23:24
 # @LastEditors: PengJie pengjieb@mail.ustc.edu.cn
 # @LastEditTime: 2025-06-26 20:27:34
 # @FilePath: /Qwen2-VL-Finetune/scripts/finetune_lora_vision.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 

# You can use 2B instead of 7B
# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

export PYTHONPATH=src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1,2,3,4
export CUDA_VISIBLE_DEVICES=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export HF_ENDPOINT=https://hf-mirror.com

GLOBAL_BATCH_SIZE=16
BATCH_PER_DEVICE=4
NUM_DEVICES=4
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))
# 112, 80, 48, 16
n_image=64 # 56 64/
n_depth=64 # 40
n_norm=64 # 24
n_flow=64 # 8
multilevel_qformer=True
image_resolution=112
out_dir=lora_vision_test_${n_image}_${n_depth}_${n_norm}_${n_flow}_token_dynamic
# If you want to tune the `embed_token` with LoRA, You need to tune `lm_head` together
# You should freeze the the merger also, becuase the merger is included in the vision_tower.

# deepspeed src/train/train_sft.py \
#     --use_liger True \
#     --lora_enable True \
#     --vision_lora True \
#     --use_dora False \
#     --lora_namespan_exclude "['lm_head', 'embed_tokens', 'm_qformer']" \
#     --lora_rank 64 \
#     --lora_alpha 64 \
#     --lora_dropout 0.05 \
#     --num_lora_modules -1 \
#     --deepspeed scripts/zero3.json \
#     --model_id $MODEL_NAME \
#     --data_path nextqa_1k/train_subset_1k_qwen.json \
#     --image_folder . \
#     --remove_unused_columns False \
#     --freeze_vision_tower True \
#     --freeze_llm True \
#     --freeze_merger True \
#     --bf16 True \
#     --fp16 False \
#     --disable_flash_attn2 False \
#     --output_dir output/$out_dir \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size $BATCH_PER_DEVICE \
#     --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
#     --image_min_pixels $((256 * 28 * 28)) \
#     --image_max_pixels $((256 * 28 * 28)) \
#     --image_resized_width $image_resolution \
#     --image_resized_height $image_resolution \
#     --learning_rate 2e-4 \
#     --weight_decay 0.1 \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --gradient_checkpointing True \
#     --report_to tensorboard \
#     --lazy_preprocess True \
#     --save_strategy "steps" \
#     --save_steps 200 \
#     --save_total_limit 10 \
#     --dataloader_num_workers 4 \
#     --n_image $n_image \
#     --n_depth $n_depth \
#     --n_norm $n_norm \
#     --n_flow $n_flow \
#     --multilevel_qformer $multilevel_qformer \

# cd output/$out_dir
# highest_checkpoint=$(ls | grep '^checkpoint-' | sed 's/^checkpoint-//' | sort -nr | head -1)
# echo "Highest checkpoint: $highest_checkpoint"
# rm non_lora_state_dict.bin
# ln -s /mnt/shared_workspace/pengjie/Qwen2-VL-Finetune/output/$out_dir/checkpoint-$highest_checkpoint/non_lora_state_dict.bin non_lora_state_dict.bin
# cd -

python src/train/eval_sft.py \
    --use_liger True \
    --lora_enable True \
    --vision_lora True \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens', 'm_qformer']" \
    --lora_rank 64 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --model_id $MODEL_NAME \
    --data_path nextqa_1k/val_1k_qwen.json \
    --model_path output/$out_dir \
    --image_folder . \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --freeze_merger True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir output/lora_vision_test \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((256 * 28 * 28)) \
    --image_max_pixels $((256 * 28 * 28)) \
    --image_resized_width $image_resolution \
    --image_resized_height $image_resolution \
    --learning_rate 2e-4 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --dataloader_num_workers 4 \
    --n_image $n_image \
    --n_depth $n_depth \
    --n_norm $n_norm \
    --n_flow $n_flow \
    --multilevel_qformer $multilevel_qformer \