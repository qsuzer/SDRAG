#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-}"
GPU_ID="${2:-0}"
BASE_DATASET_PATH="${3:-${BASE_DATASET_PATH:-}}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"

if [[ -z "${RUN_DIR}" || -z "${BASE_DATASET_PATH}" ]]; then
  echo "usage: $0 RUN_DIR [GPU_ID] BASE_DATASET_PATH" >&2
  echo "or set BASE_DATASET_PATH in the environment" >&2
  exit 1
fi
if ! [[ "${SHARD_INDEX}" =~ ^[0-9]+$ && "${SHARD_COUNT}" =~ ^[0-9]+$ ]]; then
  echo "SHARD_INDEX and SHARD_COUNT must be non-negative integers" >&2
  exit 1
fi
if (( SHARD_COUNT < 1 || SHARD_INDEX >= SHARD_COUNT )); then
  echo "Require SHARD_COUNT >= 1 and SHARD_INDEX < SHARD_COUNT" >&2
  exit 1
fi

: "${MODEL_PATH:?Set MODEL_PATH to the base instruction model path or Hugging Face id.}"

SWIFT_BIN="${SWIFT_BIN:-swift}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METRICS_SCRIPT="${METRICS_SCRIPT:-${SCRIPT_DIR}/resolve_swift_best_checkpoint.py}"
SPLIT_SCRIPT="${SPLIT_SCRIPT:-${SCRIPT_DIR}/split_single_task_data.py}"
SPLIT_DATASET_RATIO="${SPLIT_DATASET_RATIO:-0.05}"

LOGS_DIR="${RUN_DIR}/logs"
RUNS_DIR="${RUN_DIR}/runs"
DATASETS_DIR="${RUN_DIR}/datasets"
STATUS_FILE="${RUN_DIR}/status.tsv"
PLAN_FILE="${RUN_DIR}/run_plan.tsv"
SUMMARY_FILE="${RUN_DIR}/summary.tsv"
SPLIT_REPORT="${RUN_DIR}/dataset_split_report.json"

mkdir -p "${LOGS_DIR}" "${RUNS_DIR}" "${DATASETS_DIR}"
: > "${STATUS_FILE}"
printf 'task\trun_name\tdataset_path\tsplit_dataset_ratio\tlearning_rate\tlora_rank\tlora_alpha\tlora_dropout\tmax_length\tgrad_acc\tnum_train_epochs\tmax_steps\tlr_scheduler_type\twarmup_ratio\ttarget_modules\n' > "${PLAN_FILE}"
printf 'task\trun_name\trc\tactual_run_dir\tlast_eval_loss\tbest_eval_loss\tbest_eval_step\tbest_eval_token_acc\tbest_checkpoint\tlast_train_loss\tnum_eval_points\n' > "${SUMMARY_FILE}"

printf '%s\tDATASET_SPLIT_START\t%s\n' "$(date --iso-8601=seconds)" "${BASE_DATASET_PATH}" >> "${STATUS_FILE}"
"${PYTHON_BIN}" "${SPLIT_SCRIPT}" \
  --input "${BASE_DATASET_PATH}" \
  --output-dir "${DATASETS_DIR}" \
  --prefix "single_task" \
  --report "${SPLIT_REPORT}" \
  >> "${STATUS_FILE}" 2>&1
printf '%s\tDATASET_SPLIT_END\t%s\n' "$(date --iso-8601=seconds)" "${SPLIT_REPORT}" >> "${STATUS_FILE}"
printf '%s\tSHARD_INFO\tindex=%s\tcount=%s\n' "$(date --iso-8601=seconds)" "${SHARD_INDEX}" "${SHARD_COUNT}" >> "${STATUS_FILE}"

declare -A DATASET_BY_TASK=(
  ["decomposition"]="${DATASETS_DIR}/single_task_decomposition.jsonl"
  ["dependency"]="${DATASETS_DIR}/single_task_dependency.jsonl"
  ["pruning"]="${DATASETS_DIR}/single_task_pruning.jsonl"
)

declare -a CONFIG_ROWS=(
  "decomposition|decomp_alllinear_cosine_w005_r16_len1536|8e-5|16|32|0.05|1536|16|3|-1|cosine|0.05|all-linear"
  "decomposition|decomp_attn_cosine_w005_r32_len1536|6e-5|32|64|0.05|1536|16|3|-1|cosine|0.05|q_proj,k_proj,v_proj,o_proj"
  "dependency|dep_qv_cosine_w005_r16_len1536|8e-5|16|32|0.05|1536|16|3|-1|cosine|0.05|q_proj,v_proj"
  "dependency|dep_alllinear_cosine_w005_r16_len1536|5e-5|16|32|0.05|1536|16|3|-1|cosine|0.05|all-linear"
  "pruning|prune_qv_cosine_w005_r16_len1536|8e-5|16|32|0.05|1536|16|3|-1|cosine|0.05|q_proj,v_proj"
  "pruning|prune_mlp_cosine_w005_r16_len1536|8e-5|16|32|0.05|1536|16|3|-1|cosine|0.05|gate_proj,up_proj,down_proj"
)

row_idx=0
for row in "${CONFIG_ROWS[@]}"; do
  if (( row_idx % SHARD_COUNT != SHARD_INDEX )); then
    row_idx=$((row_idx + 1))
    continue
  fi

  IFS='|' read -r task run_name lr lora_rank lora_alpha lora_dropout max_length grad_acc num_train_epochs max_steps lr_scheduler warmup_ratio target_modules_csv <<< "${row}"
  dataset_path="${DATASET_BY_TASK[${task}]}"
  IFS=, read -r -a target_modules <<< "${target_modules_csv}"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${task}" "${run_name}" "${dataset_path}" "${SPLIT_DATASET_RATIO}" "${lr}" "${lora_rank}" "${lora_alpha}" "${lora_dropout}" "${max_length}" "${grad_acc}" "${num_train_epochs}" "${max_steps}" "${lr_scheduler}" "${warmup_ratio}" "${target_modules_csv}" \
    >> "${PLAN_FILE}"

  base_output_dir="${RUNS_DIR}/${task}/${run_name}"
  log_path="${LOGS_DIR}/${run_name}.log"
  mkdir -p "${base_output_dir}"

  printf '%s\tSTART\t%s\tgpu=%s\ttask=%s\t%s\n' "$(date --iso-8601=seconds)" "${run_name}" "${GPU_ID}" "${task}" "${log_path}" >> "${STATUS_FILE}"

  swift_args=(
    sft
    --model "${MODEL_PATH}"
    --dataset "${dataset_path}"
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
    --save_total_limit 2
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
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${task}" "${run_name}" "${rc}" "${actual_run_dir}" "${last_eval_loss:-}" "${best_eval_loss:-}" "${best_eval_step:-}" "${best_eval_token_acc:-}" "${best_checkpoint:-}" "${last_train_loss:-}" "${num_eval_points:-}" \
    >> "${SUMMARY_FILE}"

  row_idx=$((row_idx + 1))
done

printf '%s\tALL_DONE\tall\tdone\n' "$(date --iso-8601=seconds)" >> "${STATUS_FILE}"
