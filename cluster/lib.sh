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
