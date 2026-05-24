#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-}"
GPU_ID="${2:-0}"

if [[ -z "${RUN_DIR}" ]]; then
  echo "usage: $0 RUN_DIR [GPU_ID]" >&2
  exit 1
fi

: "${MODEL_PATH:?Set MODEL_PATH to the base instruction model path or Hugging Face id.}"
: "${DATASET_PATH:?Set DATASET_PATH to the mixed SDRAG training jsonl.}"

SWIFT_BIN="${SWIFT_BIN:-swift}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SPLIT_DATASET_RATIO="${SPLIT_DATASET_RATIO:-0.02}"
METRICS_SCRIPT="${METRICS_SCRIPT:-$(dirname "$0")/resolve_swift_best_checkpoint.py}"

LOGS_DIR="${RUN_DIR}/logs"
RUNS_DIR="${RUN_DIR}/runs"
STATUS_FILE="${RUN_DIR}/status.tsv"
PLAN_FILE="${RUN_DIR}/run_plan.tsv"
SUMMARY_FILE="${RUN_DIR}/summary.tsv"

mkdir -p "${LOGS_DIR}" "${RUNS_DIR}"
: > "${STATUS_FILE}"
printf 'run_name\tlearning_rate\tlora_rank\tlora_alpha\tlora_dropout\tmax_length\tgrad_acc\tnum_train_epochs\tmax_steps\tlr_scheduler_type\twarmup_ratio\ttarget_modules\n' > "${PLAN_FILE}"
printf 'run_name\trc\tactual_run_dir\tlast_eval_loss\tbest_eval_loss\tbest_eval_step\tbest_eval_token_acc\tbest_checkpoint\tlast_train_loss\tnum_eval_points\n' > "${SUMMARY_FILE}"

declare -a CONFIG_ROWS=(
  "alllinear_cosine_w005_r16_len1536_ep3|8e-5|16|32|0.05|1536|16|3|-1|cosine|0.05|all-linear"
  "alllinear_cosine_w003_r8_len1024|1e-4|8|16|0.05|1024|16|1|500|cosine|0.03|all-linear"
  "attn_cosine_w005_r16_len1024|1e-4|16|32|0.05|1024|16|1|500|cosine|0.05|q_proj,k_proj,v_proj,o_proj"
  "qv_cosine_w005_r16_len1536|8e-5|16|32|0.05|1536|16|1|500|cosine|0.05|q_proj,v_proj"
)

for row in "${CONFIG_ROWS[@]}"; do
  IFS='|' read -r run_name lr lora_rank lora_alpha lora_dropout max_length grad_acc num_train_epochs max_steps lr_scheduler warmup_ratio target_modules_csv <<< "${row}"
  IFS=, read -r -a target_modules <<< "${target_modules_csv}"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${run_name}" "${lr}" "${lora_rank}" "${lora_alpha}" "${lora_dropout}" "${max_length}" "${grad_acc}" "${num_train_epochs}" "${max_steps}" "${lr_scheduler}" "${warmup_ratio}" "${target_modules_csv}" \
    >> "${PLAN_FILE}"

  base_output_dir="${RUNS_DIR}/${run_name}"
  log_path="${LOGS_DIR}/${run_name}.log"
  mkdir -p "${base_output_dir}"

  printf '%s\tSTART\t%s\tgpu=%s\t%s\n' "$(date --iso-8601=seconds)" "${run_name}" "${GPU_ID}" "${log_path}" >> "${STATUS_FILE}"

  swift_args=(
    sft
    --model "${MODEL_PATH}"
    --dataset "${DATASET_PATH}"
    --split_dataset_ratio "${SPLIT_DATASET_RATIO}"
    --train_type lora
    --per_device_train_batch_size 1
    --gradient_accumulation_steps "${grad_acc}"
    --gradient_checkpointing true
    --learning_rate "${lr}"
    --lr_scheduler_type "${lr_scheduler}"
    --warmup_ratio "${warmup_ratio}"
    --num_train_epochs "${num_train_epochs}"
    --max_length "${max_length}"
    --lora_rank "${lora_rank}"
    --lora_alpha "${lora_alpha}"
    --lora_dropout "${lora_dropout}"
    --target_modules "${target_modules[@]}"
    --eval_strategy steps
    --eval_steps 100
    --save_steps 100
    --save_total_limit 3
    --logging_steps 10
    --output_dir "${base_output_dir}"
  )
  if [[ "${max_steps}" != "-1" ]]; then
    swift_args+=(--max_steps "${max_steps}")
  fi

  set +e
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  NPROC_PER_NODE=1 \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  "${SWIFT_BIN}" "${swift_args[@]}" > "${log_path}" 2>&1
  rc=$?
  set -e

  actual_run_dir=""
  if compgen -G "${base_output_dir}/v*" > /dev/null; then
    actual_run_dir="$(find "${base_output_dir}" -maxdepth 1 -mindepth 1 -type d -name 'v*' | sort | tail -n 1)"
  fi

  metrics=""
  if [[ -n "${actual_run_dir}" ]]; then
    metrics="$("${PYTHON_BIN}" "${METRICS_SCRIPT}" "${actual_run_dir}" 2>/dev/null || true)"
  fi
  IFS=$'\t' read -r last_eval_loss best_eval_loss best_eval_step best_eval_token_acc best_checkpoint last_train_loss num_eval_points <<< "${metrics}"

  printf '%s\tEND\t%s\trc=%s\t%s\n' "$(date --iso-8601=seconds)" "${run_name}" "${rc}" "${actual_run_dir}" >> "${STATUS_FILE}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${run_name}" "${rc}" "${actual_run_dir}" "${last_eval_loss:-}" "${best_eval_loss:-}" "${best_eval_step:-}" "${best_eval_token_acc:-}" "${best_checkpoint:-}" "${last_train_loss:-}" "${num_eval_points:-}" \
    >> "${SUMMARY_FILE}"
done

printf '%s\tALL_DONE\tall\tdone\n' "$(date --iso-8601=seconds)" >> "${STATUS_FILE}"
