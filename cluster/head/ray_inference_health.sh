#!/usr/bin/env bash
set -euo pipefail

# Check Ray cluster health
VLLM_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$' | head -n1)
if [[ -z "${VLLM_CONTAINER}" ]]; then
  echo "No node-* container running on this host." >&2
  exit 1
fi

docker exec "${VLLM_CONTAINER}" ray status

# Verify server health endpoint
curl -fsS http://127.0.0.1:8000/health && echo

# Monitor GPU utilization on both nodes
nvidia-smi
docker exec "${VLLM_CONTAINER}" nvidia-smi --query-gpu=memory.used,memory.total --format=csv

