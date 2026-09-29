#!/usr/bin/env bash
set -euo pipefail

# Build FlashInfer's CUTLASS fused-MoE module ahead of time, on THIS node.
#
# Why this exists as a separate step instead of letting vllm serve do it:
#
# The module is 97 nvcc translation units of heavy CUTLASS templates. vLLM only
# triggers the build on the first MoE forward -- which happens during KV-cache
# profiling, i.e. AFTER 88.63 GiB of weights are already resident. That leaves
# ~10 GiB of host headroom on GB10's unified memory, and a single `cicc` on the
# worst of these files was measured at 5.3 GiB RSS. Two in parallel is enough to
# hit earlyoom, which SIGTERMs the compiler, fails ninja, fails the worker and
# kills the engine start -- observed 2026-09-29 at object 20 of 97, ~40 minutes
# into a build, after a ~9 minute model load.
#
# Run here instead and no model is loaded, so there is ~110 GiB free rather than
# ~10 GiB. That makes the build both safe and far faster, because MAX_JOBS can
# be raised instead of throttled to 2.
#
# The result lands in ~/.cache/flashinfer on the host (bind-mounted into the Ray
# containers by cluster/*/run_cluster.sh), so it is built once and reused by
# every subsequent server start.
#
# Run on EVERY node -- the cache is per node, like the model weights.
#
# Usage:
#   bash glm/precompile-moe.sh              # MAX_JOBS=8
#   MAX_JOBS=4 bash glm/precompile-moe.sh   # if you want it gentler

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${VLLM_IMAGE:-local/vllm-ray-glm53:sm121-v11-dflash2}"
FI_CACHE="${FI_CACHE:-${HOME}/.cache/flashinfer}"
# 8 parallel jobs against ~110 GiB free is comfortable even if several land on
# the 5.3 GiB outliers simultaneously. Lower it if this node is doing other work.
JOBS="${MAX_JOBS:-8}"

mkdir -p "${FI_CACHE}"

echo "Node:     $(hostname)"
echo "Image:    ${IMAGE}"
echo "Cache:    ${FI_CACHE}"
echo "MAX_JOBS: ${JOBS}"
echo "Free RAM: $(awk '/MemAvailable/{printf "%.1f GB", $2/1048576}' /proc/meminfo)"
echo

if docker ps --format '{{.Names}}' | grep -qE '^node-[0-9]+$'; then
  echo "WARNING: a Ray container is running on this node." >&2
  echo "If a model is loaded, this build competes with it for memory -- which is" >&2
  echo "the exact failure this script exists to avoid. Stop the cluster first." >&2
  echo >&2
fi

docker run --rm --gpus all \
  -e MAX_JOBS="${JOBS}" \
  -v "${FI_CACHE}:/root/.cache/flashinfer" \
  --entrypoint /bin/bash \
  "${IMAGE}" -c '
    set -euo pipefail
    python3 - <<"PY"
import os, time
from flashinfer.fused_moe.core import gen_cutlass_fused_moe_sm120_module

t0 = time.time()
print("building cutlass fused-MoE module (sm120 path, used by SM121)...", flush=True)
# use_fast_build=False matches what vLLM requests at runtime; building with a
# different flag would produce a module vLLM does not reuse, silently wasting
# the whole compile.
mod = gen_cutlass_fused_moe_sm120_module(False)
mod.build_and_load()
print("built in %.1f min" % ((time.time() - t0) / 60), flush=True)
PY
  '

echo
SO_COUNT=$(find "${FI_CACHE}" -name '*.so' 2>/dev/null | wc -l)
echo "OK  cutlass fused-MoE built on $(hostname).  .so files in cache: ${SO_COUNT}"
echo "    Size: $(du -sh "${FI_CACHE}" 2>/dev/null | cut -f1)"
echo "    Run this on the OTHER Spark too, then serve with MOE_BACKEND=flashinfer_cutlass."
