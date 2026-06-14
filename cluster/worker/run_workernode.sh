#!/usr/bin/env bash
set -euo pipefail

# On Node 2, join as worker (single-interface fallback path; see run_workernode_2.sh
# for the 4-port data-plane version that is in use day-to-day).

# Set the interface name (same as Node 1)
export MN_IF_NAME=enp1s0f1np1

# Get Node 2's own IP address
export VLLM_HOST_IP=$(ip -4 addr show "$MN_IF_NAME" | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -n1)
if [[ -z "${VLLM_HOST_IP}" ]]; then
  echo "Could not resolve IPv4 address for $MN_IF_NAME" >&2
  exit 1
fi

# Head's IP under the legacy single-interface plan. Override via shell env:
#   HEAD_NODE_IP=10.0.1.3 bash run_workernode.sh
export HEAD_NODE_IP="${HEAD_NODE_IP:-10.0.0.3}"

# Set vLLM image
export VLLM_IMAGE=nvcr.io/nvidia/vllm:25.11-py3

echo "Worker IP: $VLLM_HOST_IP, connecting to head node at: $HEAD_NODE_IP"

bash run_cluster.sh "$VLLM_IMAGE" "$HEAD_NODE_IP" --worker ~/.cache/huggingface \
  -e VLLM_HOST_IP="$VLLM_HOST_IP" \
  -e UCX_NET_DEVICES="$MN_IF_NAME" \
  -e NCCL_SOCKET_IFNAME="$MN_IF_NAME" \
  -e OMPI_MCA_btl_tcp_if_include="$MN_IF_NAME" \
  -e GLOO_SOCKET_IFNAME="$MN_IF_NAME" \
  -e TP_SOCKET_IFNAME="$MN_IF_NAME" \
  -e RAY_memory_monitor_refresh_ms=0 \
  -e MASTER_ADDR="$HEAD_NODE_IP"

