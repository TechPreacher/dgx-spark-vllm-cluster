#!/usr/bin/env bash
set -uo pipefail
# Unit tests for the pure helpers in cluster/lib.sh.
#
# These exist because the logic they cover was wrong in ways that only showed up
# as a misleading error at bring-up time:
#   * the ray-status GPU parse was anchored on end-of-line and returned EMPTY
#     once Ray appended its placement-group annotation -- then the launcher
#     blamed the worker for not joining.
#   * counting GPUs alone cannot tell "2 nodes x 1 GB10" from "1 node x 2 GPUs",
#     so a worker accidentally started on Node 1 passed the check and vLLM put
#     181 GiB of weights on a single 128 GB host.
#   * the gpu-memory-utilization ceiling failed OPEN: awk interpolates the value
#     into program text, so "inf", "0,85" and "abc" all sailed past it.
#
# Run: bash scripts/test_cluster_lib.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../cluster/lib.sh
source "${SCRIPT_DIR}/../cluster/lib.sh"

PASS=0
FAIL=0

check_eq() {
  local desc="$1" expected="$2" actual="$3"
  if [[ "${expected}" == "${actual}" ]]; then
    PASS=$((PASS + 1)); printf '  ok    %s\n' "${desc}"
  else
    FAIL=$((FAIL + 1)); printf '  FAIL  %s\n        expected %q, got %q\n' "${desc}" "${expected}" "${actual}"
  fi
}

# Assert a predicate function's exit status, refusing to pass vacuously when the
# function does not exist (a "command not found" is also non-zero).
check_rc() {
  local desc="$1" want="$2"; shift 2
  if ! declare -F "$1" >/dev/null; then
    FAIL=$((FAIL + 1)); printf '  FAIL  %s (function %q is not defined)\n' "${desc}" "$1"; return
  fi
  "$@" >/dev/null 2>&1
  local got=$?
  if [[ "${got}" -eq "${want}" ]]; then
    PASS=$((PASS + 1)); printf '  ok    %s\n' "${desc}"
  else
    FAIL=$((FAIL + 1)); printf '  FAIL  %s (want rc %s, got %s)\n' "${desc}" "${want}" "${got}"
  fi
}

echo
echo "cluster/lib.sh unit tests"
echo

# --- ray_gpu_total_from_status ------------------------------------------------
PLAIN=' 0.0/2.0 GPU'
SINGLE=' 0.0/1.0 GPU'
# The form that broke the original end-anchored parse. vLLM itself creates a
# placement group, so this is the steady state after a serve attempt, not a
# corner case.
PGROUP=' 0.0/2.0 GPU (0.0 used of 2.0 reserved in placement groups)'
USED=' 2.0/2.0 GPU'

check_eq "ray gpus: plain two-GPU line" \
  "2" "$(ray_gpu_total_from_status "${PLAIN}")"
check_eq "ray gpus: single-GPU line" \
  "1" "$(ray_gpu_total_from_status "${SINGLE}")"
check_eq "ray gpus: placement-group annotation does not break the parse" \
  "2" "$(ray_gpu_total_from_status "${PGROUP}")"
check_eq "ray gpus: reads the TOTAL, not the used count" \
  "2" "$(ray_gpu_total_from_status "${USED}")"
check_eq "ray gpus: no GPU line at all yields 0" \
  "0" "$(ray_gpu_total_from_status ' 0.0/40.0 CPU')"

# --- ray_node_count_from_status -----------------------------------------------
TWO_NODES='Node status
---------------------------------------------------------
Active:
 1 node_abc123def456
 1 node_789fedcba321
Pending:
 (no pending nodes)
Resources
---------------------------------------------------------
Usage:
 0.0/40.0 CPU
 0.0/2.0 GPU'

ONE_NODE_TWO_GPU='Node status
---------------------------------------------------------
Active:
 1 node_abc123def456
Pending:
 (no pending nodes)
Resources
---------------------------------------------------------
Usage:
 0.0/40.0 CPU
 0.0/2.0 GPU'

check_eq "ray nodes: two active nodes counted" \
  "2" "$(ray_node_count_from_status "${TWO_NODES}")"
check_eq "ray nodes: one node reporting two GPUs counts as ONE node" \
  "1" "$(ray_node_count_from_status "${ONE_NODE_TWO_GPU}")"
check_eq "ray nodes: empty status yields 0" \
  "0" "$(ray_node_count_from_status '')"

# --- mem_util_within_ceiling --------------------------------------------------
# rc 0 = acceptable, rc 1 = refuse. Anything unparseable MUST refuse, not pass.
check_rc "mem util: 0.80 accepted"                0 mem_util_within_ceiling 0.80 0.85
check_rc "mem util: 0.85 accepted (boundary)"     0 mem_util_within_ceiling 0.85 0.85
check_rc "mem util: 0.90 refused"                 1 mem_util_within_ceiling 0.90 0.85
check_rc "mem util: 1 refused"                    1 mem_util_within_ceiling 1 0.85
check_rc "mem util: 'inf' refused (was passing)"  1 mem_util_within_ceiling inf 0.85
check_rc "mem util: '0,85' refused (was passing)" 1 mem_util_within_ceiling 0,85 0.85
check_rc "mem util: 'abc' refused (was passing)"  1 mem_util_within_ceiling abc 0.85
check_rc "mem util: empty refused"                1 mem_util_within_ceiling "" 0.85
check_rc "mem util: injection attempt refused"    1 mem_util_within_ceiling '0.5);print("x' 0.85

echo
echo "  ${PASS} passed, ${FAIL} failed"
echo
[[ "${FAIL}" -eq 0 ]]
