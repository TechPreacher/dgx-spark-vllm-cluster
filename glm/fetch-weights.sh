#!/usr/bin/env bash
set -euo pipefail

# Download the GLM checkpoint into THIS node's HuggingFace cache.
#
# Run on EVERY node before serving. With Ray TP=2 each rank loads its shard from
# its OWN node's filesystem -- the HF cache is bind-mounted per node, not shared
# -- so a checkpoint present only on the head makes rank 1 fail at load time
# with a Ray traceback that names the path but not the reason:
#
#   ray::RayWorkerProc.initialize_worker() (ip=10.0.0.2)
#   RuntimeError: Cannot find any model weights with `/root/.cache/.../snapshots/...`
#
# Why a container rather than a host-side `hf download` or an rsync from the
# head: the cache directories are created by the Ray container as root, so a
# host user cannot write into them, and rsync as that user fails on permissions
# (or silently skips). Populating the cache from inside a container writes as
# root, exactly as the serving path does, so ownership stays consistent and no
# sudo is needed.
#
# snapshot_download is resumable, so an interrupted run continues.
#
# Usage:
#   bash glm/fetch-weights.sh                 # the target checkpoint
#   MODEL_CKPT=incoai/GLM-5.3-Flash-DFlash2 bash glm/fetch-weights.sh   # drafter
#
# Roughly 181 GiB and ~30 minutes per node on a normal connection. Both nodes
# can fetch in parallel; they are independent.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../cluster/lib.sh
source "${SCRIPT_DIR}/../cluster/lib.sh"
load_env "${SCRIPT_DIR}"
: "${HF_TOKEN:?HF_TOKEN not set (expected in glm/.env -- copy glm/.env.example)}"

MODEL_CKPT="${MODEL_CKPT:-LibertAIDAI/GLM-5.3-Flash-NVFP4}"
IMAGE="${VLLM_IMAGE:-local/vllm-ray-glm53:sm121-v11-dflash2}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"

echo "Node:       $(hostname)"
echo "Checkpoint: ${MODEL_CKPT}"
echo "Cache:      ${HF_CACHE}"
echo "Image:      ${IMAGE}"
echo "Free space: $(df -h "${HF_CACHE}" | tail -1 | awk '{print $4}')"
echo

docker run --rm \
  -e HF_TOKEN="${HF_TOKEN}" \
  -e MODEL_CKPT="${MODEL_CKPT}" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  --entrypoint /bin/bash \
  "${IMAGE}" -c '
    set -euo pipefail
    python3 - <<PY
import glob, os
from huggingface_hub import snapshot_download
path = snapshot_download(os.environ["MODEL_CKPT"])
print("path:", path)
n = len(glob.glob(os.path.join(path, "*.safetensors")))
size = sum(os.path.getsize(os.path.realpath(f))
           for f in glob.glob(os.path.join(path, "*.safetensors")))
print("safetensors:", n)
print("bytes:", size)
if n == 0:
    raise SystemExit("no .safetensors in the snapshot -- refusing to report success")
PY
  '

echo
echo "OK  ${MODEL_CKPT} is present on $(hostname)."
echo "    Run this on the OTHER Spark too before serving."
