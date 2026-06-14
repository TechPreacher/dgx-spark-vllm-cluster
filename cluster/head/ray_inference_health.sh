#!/usr/bin/env bash
set -euo pipefail

# Check Ray cluster health
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib.sh
source "${SCRIPT_DIR}/../lib.sh"

VLLM_CONTAINER=$(find_ray_container)

docker exec "${VLLM_CONTAINER}" ray status

# Verify server health endpoint
curl -fsS http://127.0.0.1:8000/health && echo

# Monitor GPU utilization on both nodes
nvidia-smi
docker exec "${VLLM_CONTAINER}" nvidia-smi --query-gpu=memory.used,memory.total --format=csv

