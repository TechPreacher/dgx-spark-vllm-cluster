#!/usr/bin/env bash
set -euo pipefail

# Launch Qwen3-30B-A3B-Thinking-2507 (FP8) on the running Ray-clustered vLLM
# container on Node 1. Ray dispatches shard 2 to Node 2.
# Architecture: Qwen3MoeForCausalLM (natively supported by vLLM 0.11.0+nv25.11).
#
# To swap to community NVFP4 (better fit for Spark's native fp4 tensor cores
# but unofficial), change the model tag to:
#   MrVolts/Qwen3-30B-A3B-Thinking-2507-NVFP4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../cluster/lib.sh
source "${SCRIPT_DIR}/../cluster/lib.sh"
load_env "${SCRIPT_DIR}"
: "${VLLM_API_KEY:?VLLM_API_KEY not set (expected in .env)}"

VLLM_CONTAINER=$(find_ray_container)
echo "Using container: ${VLLM_CONTAINER}"

docker exec -it -e VLLM_API_KEY="${VLLM_API_KEY}" "${VLLM_CONTAINER}" /bin/bash -c '
  set -e
  exec vllm serve Qwen/Qwen3-30B-A3B-Thinking-2507-FP8 \
    --served-model-name qwen3_30b_thinking \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 2 \
    --max-num-seqs 8 \
    --max-model-len 131072 \
    --max-num-batched-tokens 32768 \
    --gpu-memory-utilization 0.70 \
    --enable-prefix-caching \
    --reasoning-parser deepseek_r1 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
'
