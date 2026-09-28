#!/usr/bin/env bash
set -euo pipefail

# Build the local Ray-capable vLLM image from cluster/Dockerfile.
# Run this once per node (the resulting tag is local, not registry-backed)
# and any time you bump the NGC base image.
#
# Usage:
#   bash cluster/build-image.sh                     # default tag and base
#   TAG=local/vllm-ray:dev bash cluster/build-image.sh
#   BASE_IMAGE=nvcr.io/nvidia/vllm:26.06-py3 bash cluster/build-image.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/vllm:26.05.post1-py3}"
TAG="${TAG:-local/vllm-ray:26.05.post1}"

echo "Building ${TAG} FROM ${BASE_IMAGE}..."
# Record what this was built FROM. glm/verify-image.sh gates on this label so a
# moved upstream tag cannot silently substitute different kernels underneath a
# digest-pinned base. Harmless for the Qwen/Nemotron image (records the NGC base).
docker build \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  --label "glm.base.digest=${BASE_IMAGE}" \
  -t "${TAG}" \
  "${SCRIPT_DIR}"

echo
echo "Built ${TAG}. Bring-up scripts default to this tag; override with VLLM_IMAGE."
docker images --filter "reference=${TAG}"
