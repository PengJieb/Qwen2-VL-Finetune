#!/bin/bash
###
 # @Author: PengJie pengjieb@mail.ustc.edu.cn
 # @Date: 2025-06-12 19:23:24
 # @LastEditors: PengJie pengjieb@mail.ustc.edu.cn
 # @LastEditTime: 2025-07-09 22:00:49
 # @FilePath: /Qwen2-VL-Finetune/scripts/finetune_lora_vision.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 

# You can use 2B instead of 7B
# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

export PYTHONPATH=src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export CUDA_LAUNCH_BLOCKING=1
# export HF_ENDPOINT=https://hf-mirror.com

GLOBAL_BATCH_SIZE=16
BATCH_PER_DEVICE=1
NUM_DEVICES=4
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))
PROJ_ROOT=$(pwd)
# 112, 80, 48, 16
n_image=32 # 56 64/
n_depth=32 # 40
n_norm=32 # 24
n_flow=32 # 8
multilevel_qformer=True
image_resolution=224
out_dir=lora_vision_test_${n_image}_${n_depth}_${n_norm}_${n_flow}_token_dynamic_full_crema_policy_3b_grpo_2
sft_dir=output_local/lora_vision_test_32_32_32_32_token_dynamic_full_crema_policy_fullfinetune_3b_pretrain
# output/lora_vision_test_32_32_32_32_token_dynamic_full_crema_policy_10/checkpoint-2000
# If you want to tune the `embed_token` with LoRA, You need to tune `lm_head` together
# You should freeze the the merger also, becuase the merger is included in the vision_tower.



deepspeed --master_port 29499 src/train/train_grpo.py \
    --lora_enable True \
    --vision_lora False \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens', 'm_qformer', 'prefusion']" \
    --lora_rank 64 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --deepspeed scripts/zero2.json \
    --model_id $MODEL_NAME \
    --data_path local_labels/train_subset.json \
    --image_folder . \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --freeze_merger True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir output_local/$out_dir \
    --sft_model_path $sft_dir \
    --num_train_epochs 1 \
    --num_generations 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --image_min_pixels $((256 * 28 * 28)) \
    --image_max_pixels $((256 * 28 * 28)) \
    --image_resized_width $image_resolution \
    --image_resized_height $image_resolution \
    --learning_rate 1e-5 \
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
    --max_completion_length 256 \
    --max_prompt_length 640 \
    --dataloader_num_workers 4 \
    --n_image $n_image \
    --n_depth $n_depth \
    --n_norm $n_norm \
    --n_flow $n_flow \
    --multilevel_qformer $multilevel_qformer \

cd output/$out_dir
highest_checkpoint=$(ls | grep '^checkpoint-' | sed 's/^checkpoint-//' | sort -nr | head -1)
echo "Highest checkpoint: $highest_checkpoint"
rm non_lora_state_dict.bin
ln -s $PROJ_ROOT/output_local/$out_dir/checkpoint-$highest_checkpoint/non_lora_state_dict.bin non_lora_state_dict.bin
cd -

python src/train/eval_grpo.py \
    --use_liger True \
    --lora_enable True \
    --vision_lora False \
    --use_dora True \
    --lora_namespan_exclude "['lm_head', 'embed_tokens', 'm_qformer']" \
    --lora_rank 64 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --model_id $MODEL_NAME \
    --data_path local_labels/val.json \
    --model_path output_local/$out_dir \
    --sft_model_path $sft_dir \
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