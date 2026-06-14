#!/usr/bin/env bash
set -euo pipefail

# Launch nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 on the running Ray-
# clustered vLLM container on Node 1. Ray dispatches shard 2 to Node 2 over the
# ~800 GbE data plane established by run_headnode_2.sh / run_workernode_2.sh.
#
# Reference: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
#            https://build.nvidia.com/spark/vllm
#
# Architecture: LatentMoE hybrid (Mamba-2 + MoE + Attention), 120B total / 12B
# active. Native NVFP4 quantization fits SM121 / GB10 FP4 tensor cores. With
# TP=2 across the 2-Spark Ray cluster, weights split ~half/half per node, so
# per-node memory pressure is materially lower than the single-Spark path that
# previously starved this host.
#
# ---------------------------------------------------------------------------
# Host-stability context (still applies, just less acute than single-node)
# ---------------------------------------------------------------------------
# gpt-oss-120b on a single Spark previously starved this host of memory: ICMP
# kept replying but sshd became unreachable, requiring a power cycle. The Ray
# container is launched by run_cluster.sh without a `--memory` cgroup cap, so
# this launcher CANNOT add one. Defences here are softer:
#   1. `--gpu-memory-utilization 0.75`  conservative on both nodes.
#   2. `--max-model-len 1048576`        1M tokens (model maximum). 512k was
#                                       verified stable on this hardware
#                                       (host MemAvailable ~18 GB during
#                                       inferencing); 1M roughly doubles KV
#                                       pressure on attention layers only
#                                       (hybrid model => not a 2x total).
#                                       Override via MAX_MODEL_LEN if needed.
#   3. `ENABLE_EAGER=1` opt-in          skips CUDA graph capture (memory spike).
#
# Required out-of-repo hardening (configure once on BOTH nodes):
#   * Protect sshd:
#       sudo systemctl edit ssh    # add: [Service]\nOOMScoreAdjust=-1000
#   * earlyoom:
#       sudo apt install earlyoom && sudo systemctl enable --now earlyoom
#   * External watchdog: laptop curls :8000/health every 30s, IPMI/PDU-cycles
#     on N consecutive failures. Both nodes need their own watchdog target.
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../cluster/lib.sh
source "${SCRIPT_DIR}/../cluster/lib.sh"
load_env "${SCRIPT_DIR}"
: "${VLLM_API_KEY:?VLLM_API_KEY not set (expected in nemotron/.env)}"
: "${HF_TOKEN:?HF_TOKEN not set (expected in nemotron/.env)}"

# --- Overridable knobs (env-driven) ---
MODEL_CKPT="${MODEL_CKPT:-nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4}"
SERVED_NAME="${SERVED_NAME:-nvidia/nemotron-3-super}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.75}"
TP_SIZE="${TP_SIZE:-2}"
PP_SIZE="${PP_SIZE:-1}"
PORT="${PORT:-8000}"

# vLLM flag toggles. Defaults match the HF card's recommendation for the
# upgraded cluster image (local/vllm-ray:26.05.post1, derived from
# nvcr.io/nvidia/vllm:26.05.post1-py3 via cluster/Dockerfile). If you fall back to
# an older image that rejects any of these, flip the corresponding env to 0
# (or empty string) instead of editing this file.
ENABLE_REASONING_PARSER="${ENABLE_REASONING_PARSER:-1}"
ENABLE_ASYNC_SCHEDULING="${ENABLE_ASYNC_SCHEDULING:-1}"
MOE_BACKEND="${MOE_BACKEND:-marlin}"
CUDAGRAPH_CAPTURE_SIZE="${CUDAGRAPH_CAPTURE_SIZE:-128}"

VLLM_CONTAINER=$(find_ray_container)
echo "Using container: ${VLLM_CONTAINER}"
echo "  model:               ${MODEL_CKPT}"
echo "  TP / PP:             ${TP_SIZE} / ${PP_SIZE}"
echo "  max-model-len:       ${MAX_MODEL_LEN}"
echo "  gpu-mem-util:        ${GPU_MEM_UTIL}"
echo "  MTP spec decode:     ${ENABLE_MTP:-0}"
echo "  enforce-eager:       ${ENABLE_EAGER:-0}"
echo "  reasoning parser:    ${ENABLE_REASONING_PARSER}"
echo "  async scheduling:    ${ENABLE_ASYNC_SCHEDULING}"
echo "  moe backend:         ${MOE_BACKEND:-(default)}"
echo "  cudagraph capture:   ${CUDAGRAPH_CAPTURE_SIZE:-(default)}"
echo "  port:                ${PORT}"

# Optional MTP speculative decoding (off by default; HF DGX Spark example
# uses {"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}).
# Note: spec decoding across TP=2 over the inter-node link adds latency; verify
# end-to-end throughput before keeping it on.
SPEC_FLAG=""
if [[ "${ENABLE_MTP:-0}" == "1" ]]; then
  SPEC_FLAG='--speculative_config {"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}'
fi

# Optional eager mode (disables CUDA graphs; trades speed for stability if
# graph capture spikes memory on first inference).
EAGER_FLAG=""
if [[ "${ENABLE_EAGER:-0}" == "1" ]]; then
  EAGER_FLAG="--enforce-eager"
fi

# Sanity-check that the head Ray container actually has the NVFP4 vars in its
# env. If missing, the user forgot to source nemotron/cluster-env.sh before
# bringing the cluster up -- rank 1 on the worker won't see them either, and
# the run will fail at the first FP4 matmul or allreduce. Exit early with a
# clear message rather than letting it crash inside vLLM.
MISSING_VARS=$(docker exec "${VLLM_CONTAINER}" /bin/bash -c '
  set -u
  missing=""
  for V in VLLM_NVFP4_GEMM_BACKEND VLLM_FLASHINFER_ALLREDUCE_BACKEND VLLM_USE_FLASHINFER_MOE_FP4 VLLM_ALLOW_LONG_MAX_MODEL_LEN; do
    [[ -z "${!V:-}" ]] && missing="${missing} $V"
  done
  echo "${missing}"
' | xargs)
if [[ -n "${MISSING_VARS}" ]]; then
  cat >&2 <<EOF
ERROR: Required NVFP4 env vars are not set inside the Ray container:
  ${MISSING_VARS}

These must be present at container START time on BOTH nodes; they cannot be
added now via docker exec because Ray-spawned rank-1 workers on the worker
node would still be missing them. Tear the cluster down and bring it back up
after sourcing nemotron/cluster-env.sh on each node:

  source nemotron/cluster-env.sh
  cd cluster/head && bash run_headnode_2.sh        # on Node 1
  source nemotron/cluster-env.sh
  cd cluster/worker && bash run_workernode_2.sh    # on Node 2
EOF
  exit 1
fi

docker exec -it \
  -e VLLM_API_KEY="${VLLM_API_KEY}" \
  -e HF_TOKEN="${HF_TOKEN}" \
  -e MODEL_CKPT="${MODEL_CKPT}" \
  -e SERVED_NAME="${SERVED_NAME}" \
  -e MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
  -e GPU_MEM_UTIL="${GPU_MEM_UTIL}" \
  -e TP_SIZE="${TP_SIZE}" \
  -e PP_SIZE="${PP_SIZE}" \
  -e PORT="${PORT}" \
  -e SPEC_FLAG="${SPEC_FLAG}" \
  -e EAGER_FLAG="${EAGER_FLAG}" \
  -e ENABLE_REASONING_PARSER="${ENABLE_REASONING_PARSER}" \
  -e ENABLE_ASYNC_SCHEDULING="${ENABLE_ASYNC_SCHEDULING}" \
  -e MOE_BACKEND="${MOE_BACKEND}" \
  -e CUDAGRAPH_CAPTURE_SIZE="${CUDAGRAPH_CAPTURE_SIZE}" \
  -e MAMBA_SSM_DTYPE="${MAMBA_SSM_DTYPE:-auto}" \
  "${VLLM_CONTAINER}" /bin/bash -c '
    set -euo pipefail

    # Build optional flag list based on what the installed vLLM supports.
    OPT_FLAGS=()
    if [[ "${ENABLE_REASONING_PARSER}" == "1" ]]; then
      # Fetch the plugin once; cached in HF cache so it survives container
      # restarts (~/.cache/huggingface is bind-mounted by run_cluster.sh).
      PARSER=/root/.cache/huggingface/super_v3_reasoning_parser.py
      if [[ ! -f "${PARSER}" ]]; then
        echo "Fetching reasoning parser plugin..."
        curl -fsSL -o "${PARSER}" \
          "https://huggingface.co/${MODEL_CKPT}/raw/main/super_v3_reasoning_parser.py"
      fi
      OPT_FLAGS+=(--reasoning-parser-plugin "${PARSER}" --reasoning-parser super_v3)
    fi
    [[ "${ENABLE_ASYNC_SCHEDULING}" == "1" ]] && OPT_FLAGS+=(--async-scheduling)
    [[ -n "${MOE_BACKEND}" ]]                && OPT_FLAGS+=(--moe-backend "${MOE_BACKEND}")
    [[ -n "${CUDAGRAPH_CAPTURE_SIZE}" ]]     && OPT_FLAGS+=(--max-cudagraph-capture-size "${CUDAGRAPH_CAPTURE_SIZE}")

    # shellcheck disable=SC2086
    exec vllm serve "${MODEL_CKPT}" \
      --served-model-name "${SERVED_NAME}" \
      --host 0.0.0.0 \
      --port "${PORT}" \
      --tensor-parallel-size "${TP_SIZE}" \
      --pipeline-parallel-size "${PP_SIZE}" \
      --data-parallel-size 1 \
      --quantization fp4 \
      --dtype auto \
      --kv-cache-dtype fp8 \
      --mamba-ssm-cache-dtype "${MAMBA_SSM_DTYPE}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --gpu-memory-utilization "${GPU_MEM_UTIL}" \
      --enable-chunked-prefill \
      --trust-remote-code \
      --enable-auto-tool-choice \
      --tool-call-parser qwen3_coder \
      "${OPT_FLAGS[@]}" \
      ${EAGER_FLAG} \
      ${SPEC_FLAG}
  '
