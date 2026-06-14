#!/usr/bin/env bash
set -euo pipefail

# Launch Qwen3.5-122B-A10B-FP8 (thinking) on the running Ray-clustered vLLM
# container on Node 1. Ray dispatches shard 2 to Node 2.
#
# Architecture: hybrid Gated DeltaNet + Gated Attention MoE (Qwen3-Next family),
# 122B total / 10B active, 256 experts (8 routed + 1 shared), native 262k ctx.
# Multimodal (vision) — text-only inference here.
#
# Memory budget: FP8 weights ~125 GB → ~63 GB/node at TP=2 on 128 GB unified
# memory. gpu-memory-utilization bumped 0.70 → 0.85 and max-model-len cut
# 131072 → 65536 to keep room for KV cache + CUDA graphs. The hybrid
# attention keeps KV small (only 12 of 48 layers are full-attention), so 64k
# is conservative — raise toward 131072 if memory holds after warmup.
#
# Parser changes vs. the 30B Thinking script:
#   --reasoning-parser  deepseek_r1 -> qwen3
#   --tool-call-parser  hermes      -> qwen3_coder
#
# No NVFP4 variant of this 122B model is published upstream (as of 2026-06).
# To swap back to the 30B Thinking model, restore launch.backup.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../cluster/lib.sh
source "${SCRIPT_DIR}/../cluster/lib.sh"
load_env "${SCRIPT_DIR}"
: "${VLLM_API_KEY:?VLLM_API_KEY not set (expected in .env)}"

VLLM_CONTAINER=$(find_ray_container)
echo "Using container: ${VLLM_CONTAINER}"

docker exec -it -e VLLM_API_KEY="${VLLM_API_KEY}" "${VLLM_CONTAINER}" /bin/bash -c '
  set -e
  exec vllm serve Qwen/Qwen3.5-122B-A10B-FP8 \
    --served-model-name qwen35_122b_thinking \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 2 \
    --max-num-seqs 4 \
    --max-model-len 65536 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.85 \
    --enable-prefix-caching \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --trust-remote-code
'
