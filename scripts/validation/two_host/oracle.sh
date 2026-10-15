#!/bin/bash
# Independent Transformers reference for one prompt/output size, under resource_guard.py.
# Usage: oracle.sh ENV_FILE PROMPT_TOKENS OUTPUT_TOKENS
# Writes RUN_DIR/inputs/reference-<prompt>-<output>-f16.json with the isolated
# reference environment (REFERENCE_PYTHON), eager attention and no TF32.
ENV_FILE=${1:-}
PROMPT_TOKENS=${2:-}
OUTPUT_TOKENS=${3:-}
. "$(dirname "$0")/lib.sh"
require RUN_DIR SOURCE_DIR CUDA_MODEL_ROOT REFERENCE_PYTHON ORACLE_GPU_LAYERS GUARD_SECONDS
[ -n "$PROMPT_TOKENS" ] && [ -n "$OUTPUT_TOKENS" ] || {
  echo "usage: $0 ENV_FILE PROMPT_TOKENS OUTPUT_TOKENS" >&2
  exit 2
}
stem="oracle-$PROMPT_TOKENS-$OUTPUT_TOKENS"
LOG_FILE=$RUN_DIR/$stem.driver.log

cd "$SOURCE_DIR"
{
  echo "# prelaunch $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  free -b | head -3
  nvidia-smi --query-gpu=memory.used,memory.free --format=csv
  echo "gpu processes: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
} > "$RUN_DIR/$stem.prelaunch.txt"
"$SOURCE_DIR/.venv/bin/python" scripts/validation/resource_guard.py \
  --seconds "$GUARD_SECONDS" --record "$RUN_DIR/$stem.guard.jsonl" -- \
  "$REFERENCE_PYTHON" scripts/validation/checkpoint_reference.py \
  "$CUDA_MODEL_ROOT" "$RUN_DIR/inputs/reference-$PROMPT_TOKENS-$OUTPUT_TOKENS-f16.json" \
  --plain --dtype f16 --gpu-layers "$ORACLE_GPU_LAYERS" \
  --prompt-tokens "$PROMPT_TOKENS" --output-tokens "$OUTPUT_TOKENS" 2>&1 | tee -a "$LOG_FILE"
