#!/usr/bin/env bash
set -euo pipefail

# On Node 1, start head node

# The 2 physical 200G ConnectX-7 ports appear as 4 PCIe functions (f0/f1 halves
# of each port). Cold boot brings all four up (full ~2x200G); a warm reboot
# sheds the f1 half of each port. See select_up_dataplane in cluster/lib.sh.
#
# Control plane: pin to enp1s0f0np0 -- the f0 half that is up after *both* cold
# and warm boots -- so Ray/VLLM_HOST_IP/MASTER_ADDR always have an interface to
# bind to (enp1s0f1np1, the old control IF, is gone after a warm reboot).
# Control traffic is coordination-only; it does not gate NCCL data-plane bandwidth.
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
export VLLM_IMAGE="${VLLM_IMAGE:-local/vllm-ray:26.05.post1}"

echo "Primary (control) interface: $PRIMARY_IF  IP: $VLLM_HOST_IP"
echo "Data-plane interfaces:       $DATA_IFS"

# Optional verbose NCCL logging. Set in shell env before launching, e.g.:
#   NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET bash run_headnode_2.sh
# Default (unset) keeps logs quiet for steady-state operation.
NCCL_DEBUG_ARGS=()
[[ -n "${NCCL_DEBUG:-}" ]]        && NCCL_DEBUG_ARGS+=(-e "NCCL_DEBUG=${NCCL_DEBUG}")
[[ -n "${NCCL_DEBUG_SUBSYS:-}" ]] && NCCL_DEBUG_ARGS+=(-e "NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS}")

# Model-specific env-var passthrough. Set VLLM_FORWARD_VARS to a space-separated
# list of variable names that should be forwarded into the Ray container's env
# at start time. Cross-node vLLM workers spawned by Ray inherit this container
# env (they cannot pick up vars set later via `docker exec -e`), so model
# launchers that depend on these (e.g. Nemotron NVFP4) require the cluster to
# be brought up with them already exported. See nemotron/cluster-env.sh.
EXTRA_ENV_ARGS=()
for V in ${VLLM_FORWARD_VARS:-}; do
  [[ -n "${!V:-}" ]] && EXTRA_ENV_ARGS+=(-e "$V=${!V}")
done

bash run_cluster.sh "$VLLM_IMAGE" "$VLLM_HOST_IP" --head ~/.cache/huggingface \
  -e VLLM_HOST_IP="$VLLM_HOST_IP" \
  -e UCX_NET_DEVICES="$UCX_DEVS" \
  -e NCCL_SOCKET_IFNAME="$DATA_IFS" \
  -e NCCL_IB_HCA="$RDMA_HCAS" \
  -e NCCL_CROSS_NIC=1 \
  -e OMPI_MCA_btl_tcp_if_include="$DATA_IFS" \
  -e GLOO_SOCKET_IFNAME="$DATA_IFS" \
  -e TP_SOCKET_IFNAME="$PRIMARY_IF" \
  -e RAY_memory_monitor_refresh_ms=0 \
  -e MASTER_ADDR="$VLLM_HOST_IP" \
  "${NCCL_DEBUG_ARGS[@]}" \
  "${EXTRA_ENV_ARGS[@]}"
