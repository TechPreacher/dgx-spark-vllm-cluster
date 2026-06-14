# Check Ray cluster health
export VLLM_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$')
docker exec $VLLM_CONTAINER ray status

# Verify server health endpoint
curl http://127.0.0.1:8000/health

# Monitor GPU utilization on both nodes
nvidia-smi
export VLLM_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$')
docker exec $VLLM_CONTAINER nvidia-smi --query-gpu=memory.used,memory.total --format=csv

