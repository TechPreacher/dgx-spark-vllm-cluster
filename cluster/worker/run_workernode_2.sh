#!/usr/bin/env bash
set -euo pipefail

# On Node 2, join as worker

# The 2 physical 200G ConnectX-7 ports appear as 4 PCIe functions (f0/f1 halves
# of each port). Cold boot brings all four up (full ~2x200G); a warm reboot
# sheds the f1 half of each port. See select_up_dataplane in cluster/lib.sh.
# Must match the control interface and selection logic used on Node 1.
#
# Control plane: pin to enp1s0f0np0 -- the f0 half that is up after *both* cold
# and warm boots -- so Ray/VLLM_HOST_IP/MASTER_ADDR always have an interface to
# bind to. Control traffic is coordination-only; it does not gate NCCL bandwidth.
export PRIMARY_IF=enp1s0f0np0
# Data plane: enumerate the CX7 links that currently have carrier and export
# DATA_IFS / RDMA_HCAS / UCX_DEVS -- all four HCAs after a cold boot, only the
# live ones after a warm reboot (never a down HCA, which can stall NCCL init).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib.sh
source "${SCRIPT_DIR}/../lib.sh"
select_up_dataplane

export VLLM_HOST_IP=$(ip -4 addr show "$PRIMARY_IF" | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -n1)
if [[ -z "${VLLM_HOST_IP}" ]]; then
  echo "Could not resolve IPv4 address for $PRIMARY_IF" >&2
  exit 1
fi

# Head's control IP (its enp1s0f0np0 address). Override via shell env if it moves:
#   HEAD_NODE_IP=10.0.0.x bash run_workernode_2.sh
export HEAD_NODE_IP="${HEAD_NODE_IP:-10.0.0.1}"

export VLLM_IMAGE="${VLLM_IMAGE:-local/vllm-ray:26.05.post1}"

echo "Primary (control) interface: $PRIMARY_IF  IP: $VLLM_HOST_IP"
echo "Data-plane interfaces:       $DATA_IFS"
echo "Connecting to head node at:  $HEAD_NODE_IP"

# Optional verbose NCCL logging. Set in shell env before launching, e.g.:
#   NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET bash run_workernode_2.sh
# Default (unset) keeps logs quiet for steady-state operation.
NCCL_DEBUG_ARGS=()
[[ -n "${NCCL_DEBUG:-}" ]]        && NCCL_DEBUG_ARGS+=(-e "NCCL_DEBUG=${NCCL_DEBUG}")
[[ -n "${NCCL_DEBUG_SUBSYS:-}" ]] && NCCL_DEBUG_ARGS+=(-e "NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS}")

# Model-specific env-var passthrough. Must match the head node's VLLM_FORWARD_VARS
# (or be a superset of it) -- both ends need the same vLLM runtime flags for
# rank 0 and rank 1 to agree on kernel backends and collective transports.
# See nemotron/cluster-env.sh.
EXTRA_ENV_ARGS=()
for V in ${VLLM_FORWARD_VARS:-}; do
  [[ -n "${!V:-}" ]] && EXTRA_ENV_ARGS+=(-e "$V=${!V}")
done

bash run_cluster.sh "$VLLM_IMAGE" "$HEAD_NODE_IP" --worker ~/.cache/huggingface \
  -e VLLM_HOST_IP="$VLLM_HOST_IP" \
  -e UCX_NET_DEVICES="$UCX_DEVS" \
  -e NCCL_SOCKET_IFNAME="$DATA_IFS" \
  -e NCCL_IB_HCA="$RDMA_HCAS" \
  -e NCCL_CROSS_NIC=1 \
  -e OMPI_MCA_btl_tcp_if_include="$DATA_IFS" \
  -e GLOO_SOCKET_IFNAME="$DATA_IFS" \
  -e TP_SOCKET_IFNAME="$PRIMARY_IF" \
  -e RAY_memory_monitor_refresh_ms=0 \
  -e MASTER_ADDR="$HEAD_NODE_IP" \
  "${NCCL_DEBUG_ARGS[@]}" \
  "${EXTRA_ENV_ARGS[@]}"
