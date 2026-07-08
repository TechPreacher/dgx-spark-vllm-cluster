#!/usr/bin/env bash
# Shared helpers for the model launchers and the health-check script.
# Source this file; do not execute it.
#
#   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   source "${SCRIPT_DIR}/../cluster/lib.sh"
#   load_env "${SCRIPT_DIR}"
#   VLLM_CONTAINER=$(find_ray_container)

# Source <dir>/.env if it exists, auto-exporting every var defined there.
# No-op if the file is absent (the launcher's : "${VAR:?...}" guards will catch
# the missing required ones).
load_env() {
  local dir="$1"
  if [[ -f "${dir}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${dir}/.env"
    set +a
  fi
}

# Echo the name of the local Ray vLLM container (the one started by
# cluster/{head,worker}/run_cluster.sh). Exits 1 with a clear message if none
# is running, so callers can `VLLM_CONTAINER=$(find_ray_container)` and trust
# the result.
find_ray_container() {
  local name
  name=$(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$' | head -n1)
  if [[ -z "${name}" ]]; then
    echo "No node-* container running on this host. Start the Ray cluster first (cluster/head/run_headnode_2.sh or cluster/worker/run_workernode_2.sh)." >&2
    exit 1
  fi
  echo "${name}"
}

# --- DGX Spark ConnectX-7 data-plane selection -------------------------------
# The Spark exposes its 2 physical 200G QSFP ports as 4 PCIe functions: f0/f1
# are the two x4-PCIe (multi-host) halves of the *same* physical port
# (0000:01 = port A -> enp1s0f0np0/enp1s0f1np1, 0002:01 = port B ->
# enP2p1s0f0np0/enP2p1s0f1np1). A *cold* boot brings all four up (full ~2x200G).
# A *warm* reboot sheds the f1 half of each port: the CX7 firmware latches an
# "insufficient power on the PCIe slot (27W)" state that only a real power cycle
# (AC removed, PCIe capacitors discharged) clears -- no admin bounce, devlink
# reload, or FLR recovers it. So the usable link set depends on boot type:
# 4 links after a cold boot, 2 links (the f0 halves) after a warm reboot.
#
# select_up_dataplane enumerates only the CX7 links that currently have carrier
# and exports DATA_IFS / RDMA_HCAS / UCX_DEVS from them, so NCCL_IB_HCA and UCX
# are never handed a down HCA (a down device in NCCL_IB_HCA can stall collective
# init). Fixed netdev -> RoCE HCA mapping for this hardware:
_CX7_NETDEVS=(enp1s0f0np0 enp1s0f1np1 enP2p1s0f0np0 enP2p1s0f1np1)
_CX7_HCAS=(   rocep1s0f0  rocep1s0f1  roceP2p1s0f0  roceP2p1s0f1)

# Populate DATA_IFS (netdevs, comma-joined), RDMA_HCAS (NCCL_IB_HCA), and
# UCX_DEVS (RoCE dev:port for UCX_NET_DEVICES) from the carrier-up CX7 links.
# Returns non-zero if none are up, so the caller's `set -e` aborts the bring-up.
select_up_dataplane() {
  local up_ifs=() up_hcas=() up_ucx=() i dev carrier
  for i in "${!_CX7_NETDEVS[@]}"; do
    dev="${_CX7_NETDEVS[$i]}"
    carrier=$(cat "/sys/class/net/${dev}/carrier" 2>/dev/null || echo 0)
    if [[ "${carrier}" == "1" ]]; then
      up_ifs+=("${dev}")
      up_hcas+=("${_CX7_HCAS[$i]}")
      up_ucx+=("${_CX7_HCAS[$i]}:1")
    fi
  done
  if [[ "${#up_ifs[@]}" -eq 0 ]]; then
    echo "No ConnectX-7 data-plane link has carrier (checked: ${_CX7_NETDEVS[*]})." >&2
    echo "Cold-boot the node (power off, unplug AC ~30s, replug) to bring the links up." >&2
    return 1
  fi
  local IFS=,
  DATA_IFS="${up_ifs[*]}"
  RDMA_HCAS="${up_hcas[*]}"
  UCX_DEVS="${up_ucx[*]}"
  export DATA_IFS RDMA_HCAS UCX_DEVS
  echo "ConnectX-7 data-plane links up: ${#up_ifs[@]}/4  (${DATA_IFS})"
}
