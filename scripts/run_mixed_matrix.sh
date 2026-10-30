#!/usr/bin/env bash
# Parameterized convenience entrypoint. Does not start/stop a model service.
set -euo pipefail
: "${BASE_URL:?Set BASE_URL, without /v1}"
: "${MODEL:?Set MODEL to the served model name}"
: "${DATASET:?Set DATASET to an existing JSONL file}"
DP_SIZE=${DP_SIZE:-1}
BATCH_SIZES=${BATCH_SIZES:-"8 16 32 64"}
RESULT_ROOT=${RESULT_ROOT:-results}
RUN_LABEL=${RUN_LABEL:-baseline}
if [[ ! "$RUN_LABEL" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo 'RUN_LABEL may contain letters, digits, hyphen and underscore only' >&2
  exit 2
fi
read -r -a batch_list <<< "$BATCH_SIZES"
metrics_url=${METRICS_URL:-"${BASE_URL%/}/metrics"}
result_dir="$RESULT_ROOT/$(date -u +%Y%m%dT%H%M%SZ)-${RUN_LABEL}-$$"
printf 'Result directory: %s\n' "$result_dir"
exec python -m llm_decode_bench sweep \
  --base-url "$BASE_URL" --backend sglang --model "$MODEL" \
  --dataset "$DATASET" --dp-size "$DP_SIZE" --batch-sizes "${batch_list[@]}" \
  --route-dp --ignore-eos --warmup --output-tokens "${OUTPUT_TOKENS:-4096}" \
  --steady-mode metrics --metrics-url "$metrics_url" \
  --repeats "${REPEATS:-1}" --label "$RUN_LABEL" --output "$result_dir" "$@"
