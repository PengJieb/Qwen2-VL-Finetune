#!/usr/bin/env bash

# ===== Model / Env =====
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
export HOME=/playpen/pengjie_xinyu
export PYTHONPATH="src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export DS_PORT=${DS_PORT:-25901}

# ===== Training globals =====
GLOBAL_BATCH_SIZE=16
BATCH_PER_DEVICE=1
NUM_DEVICES=4
GRAD_ACCUM_STEPS=$(( GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES) ))

# Vision/token config (constant)
n_image=32
n_depth=32
n_norm=32
n_flow=32
image_resolution=448

# Base outdir to match your existing structure
BASE_OUTDIR_ROOT="train_with_best_config_output"

# Best-average base knobs for '-' fields
modality_ranker="attention"
learnable_attention="False"
n_prefusion_layers="3"
only_self_attention="False"

# ===== Only ave=0.0 runs with your original r.* tags =====
# (Left side = r#, right side = varying knobs)
declare -a RUNS=(
  "r4  token_pruning_method=qformer           discrete_token_number=False token_distribution=top-3     parameterized_pooling=True"
  "r5  token_pruning_method=linear_pooling    discrete_token_number=True  token_distribution=top-1     parameterized_pooling=True"
  "r6  token_pruning_method=linear_pooling    discrete_token_number=True  token_distribution=top-2     parameterized_pooling=True"
  "r7  token_pruning_method=linear_pooling    discrete_token_number=True  token_distribution=top-3     parameterized_pooling=True"
  "r8  token_pruning_method=linear_pooling    discrete_token_number=False token_distribution=top-3     parameterized_pooling=True"
  "r9  token_pruning_method=linear_pooling    discrete_token_number=True  token_distribution=top-1     parameterized_pooling=False"
  "r10 token_pruning_method=linear_pooling    discrete_token_number=True  token_distribution=top-2     parameterized_pooling=False"
  "r12 token_pruning_method=linear_pooling    discrete_token_number=True  token_distribution=decrease  parameterized_pooling=False"
  "r13 token_pruning_method=linear_pooling    discrete_token_number=False token_distribution=top-3     parameterized_pooling=False"
  "r14 token_pruning_method=linear_pooling    discrete_token_number=False token_distribution=top-1     parameterized_pooling=True"
  "r15 token_pruning_method=cosine_similarity discrete_token_number=True  token_distribution=top-1     parameterized_pooling=True"
  "r17 token_pruning_method=cosine_similarity discrete_token_number=True  token_distribution=top-3     parameterized_pooling=True"
  "r18 token_pruning_method=cosine_similarity discrete_token_number=True  token_distribution=decrease  parameterized_pooling=True"
  "r19 token_pruning_method=cosine_similarity discrete_token_number=False token_distribution=decrease  parameterized_pooling=True"
)

# helper: build OUT_TAG exactly like your folders
build_out_tag () {
  local tp="$1" disc="$2" td="$3" pp="$4"
  printf "mr-%s_la-%s_osa-%s_npre-%s_tp-%s_disc-%s_td-%s_pp-%s" \
    "$modality_ranker" "$learnable_attention" "$only_self_attention" "$n_prefusion_layers" \
    "$tp" "$disc" "$td" "$pp"
}

for line in "${RUNS[@]}"; do
  # split the "rN ..." record
  read -r rtag kvs <<< "$line"
  read -r -a KV <<< "$kvs"

  # extract values
  tp="${KV[0]#token_pruning_method=}"
  disc="${KV[1]#discrete_token_number=}"
  td="${KV[2]#token_distribution=}"
  pp="${KV[3]#parameterized_pooling=}"

  OUT_TAG="${rtag}_$(build_out_tag "$tp" "$disc" "$td" "$pp")"
  OUT_DIR="${BASE_OUTDIR_ROOT}/${OUT_TAG}"
  mkdir -p "${OUT_DIR}"

  echo "======== ${rtag} (ave=0) ========"
  echo "modality_ranker=${modality_ranker}  learnable_attention=${learnable_attention}"
  echo "n_prefusion_layers=${n_prefusion_layers}  only_self_attention=${only_self_attention}"
  echo "token_pruning_method=${tp}  discrete_token_number=${disc}"
  echo "token_distribution=${td}  parameterized_pooling=${pp}"
  echo "Output -> ${OUT_DIR}"
  echo "================================="

  deepspeed --master_port "${DS_PORT}" src/train/train_sft.py \
    --use_liger True \
    --lora_enable True \
    --vision_lora False \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens', 'm_qformer']" \
    --lora_rank 64 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --deepspeed scripts/zero2.json \
    --model_id "${MODEL_NAME}" \
    --data_path local_labels/train_subset.json \
    --image_folder . \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --freeze_merger True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir "${OUT_DIR}" \
    --num_train_epochs 2 \
    --per_device_train_batch_size "${BATCH_PER_DEVICE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
    --image_min_pixels $((256 * 28 * 28)) \
    --image_max_pixels $((256 * 28 * 28)) \
    --image_resized_width  "${image_resolution}" \
    --image_resized_height "${image_resolution}" \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 120 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy steps \
    --save_steps 2000 \
    --save_total_limit 1 \
    --dataloader_num_workers 8 \
    --n_image "${n_image}" \
    --n_depth "${n_depth}" \
    --n_norm  "${n_norm}" \
    --n_flow  "${n_flow}" \
    --modality_ranker "${modality_ranker}" \
    --learnable_attention "${learnable_attention}" \
    --n_prefusion_layers "${n_prefusion_layers}" \
    --only_self_attention "${only_self_attention}" \
    --token_pruning_method "${tp}" \
    --discrete_token_number "${disc}" \
    --token_distribution "${td}" \
    --parameterized_pooling "${pp}"

  echo
done

echo "All ave=0 runs (with original r.* tags) finished."