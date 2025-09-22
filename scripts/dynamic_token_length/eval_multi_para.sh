#!/bin/bash

###
# Batch runner: loop over train_with_best_config_output/* and create matching eval_output/*
# Runs 4 jobs at a time on GPUs 0,1,2,3.
###

# ====== static config you had ======
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
export HOME=/playpen/pengjie_xinyu
export PYTHONPATH=src:$PYTHONPATH
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# ---- batch/accum for single-GPU runs ----
GLOBAL_BATCH_SIZE=16
BATCH_PER_DEVICE=2
NUM_DEVICES=1                           # single GPU per process
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

# Vision / tokens
n_image=32
n_depth=32
n_norm=32
n_flow=32
multilevel_qformer=False
multilevel_mlp=True
image_resolution=448

# ====== helper to launch one job on a specific GPU ======
run_one () {
  local SRC_DIR="$1"
  local GPU_ID="$2"
  local FOLDER_NAME
  FOLDER_NAME=$(basename "${SRC_DIR}")
  local OUT_DIR="eval_output/${FOLDER_NAME}"

  mkdir -p "${OUT_DIR}"

  echo "[GPU ${GPU_ID}] train_with_best_config_output/${FOLDER_NAME} -> ${OUT_DIR}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python src/train/eval_sft.py \
      --use_liger True \
      --lora_enable True \
      --vision_lora False \
      --use_dora False \
      --lora_namespan_exclude "['lm_head', 'embed_tokens', 'm_qformer']" \
      --lora_rank 64 \
      --lora_alpha 64 \
      --lora_dropout 0.05 \
      --num_lora_modules -1 \
      --model_id "${MODEL_NAME}" \
      --data_path local_labels/grpo_val_subset.json \
      --model_path "${SRC_DIR}" \
      --image_folder . \
      --remove_unused_columns False \
      --freeze_vision_tower True \
      --freeze_llm True \
      --freeze_merger True \
      --bf16 True \
      --fp16 False \
      --disable_flash_attn2 False \
      --output_dir "${OUT_DIR}" \
      --num_train_epochs 1 \
      --per_device_train_batch_size ${BATCH_PER_DEVICE} \
      --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
      --image_min_pixels $((256 * 28 * 28)) \
      --image_max_pixels $((256 * 28 * 28)) \
      --image_resized_width ${image_resolution} \
      --image_resized_height ${image_resolution} \
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
      --n_image ${n_image} \
      --n_depth ${n_depth} \
      --n_norm ${n_norm} \
      --n_flow ${n_flow} \
      --multilevel_qformer ${multilevel_qformer} \
      --multilevel_mlp ${multilevel_mlp} \
  > "${OUT_DIR}/run.log" 2>&1 &
}

# ====== collect source folders under train_with_best_config_output/ ======
mapfile -t ALL_SRCS < <(find train_with_best_config_output -mindepth 1 -maxdepth 1 -type d | sort)

if [[ ${#ALL_SRCS[@]} -eq 0 ]]; then
  echo "No subfolders found under train_with_best_config_output/. Nothing to run."
  exit 0
fi

# ====== run in batches of 4 on GPUs 0,1,2,3 ======
GPU_LIST=(0 1 2 3 4 5 6 7)
batch_size=${#GPU_LIST[@]}

i=0
while [[ $i -lt ${#ALL_SRCS[@]} ]]; do
  echo "=== Launching batch starting at index $i ==="
  pids=()

  for gi in "${!GPU_LIST[@]}"; do
    idx=$((i + gi))
    [[ $idx -ge ${#ALL_SRCS[@]} ]] && break
    src_dir="${ALL_SRCS[$idx]}"
    gpu="${GPU_LIST[$gi]}"
    run_one "${src_dir}" "${gpu}"
    pids+=($!)
  done

  # Wait for this batch to finish
  for pid in "${pids[@]}"; do
    wait "$pid"
  done

  echo "=== Batch finished (up to index $((i + batch_size - 1))) ==="
  i=$((i + batch_size))
done

echo "All jobs completed."