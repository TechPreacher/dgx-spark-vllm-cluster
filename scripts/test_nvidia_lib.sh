#!/usr/bin/env bash
set -uo pipefail
# Unit tests for the pure string helpers in nvidia_lib.sh.
# These helpers are deliberately pure (no dpkg, no /lib/modules) so they can be
# tested without root and without mutating the host.
#
# Run: bash scripts/test_nvidia_lib.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=nvidia_lib.sh
source "${SCRIPT_DIR}/nvidia_lib.sh"

PASS=0
FAIL=0

check_eq() {
  local desc="$1" expected="$2" actual="$3"
  if [[ "${expected}" == "${actual}" ]]; then
    PASS=$((PASS + 1))
    printf '  ok    %s\n' "${desc}"
  else
    FAIL=$((FAIL + 1))
    printf '  FAIL  %s\n        expected %q, got %q\n' "${desc}" "${expected}" "${actual}"
  fi
}

# Assert non-zero exit -- but only from a function that actually exists.
# Without the existence guard this passes vacuously when the implementation is
# absent, because "command not found" is also a non-zero exit. That is exactly
# what it did on the first RED run of this file.
check_rc_nonzero() {
  local desc="$1"; shift
  if ! declare -F "$1" >/dev/null; then
    FAIL=$((FAIL + 1))
    printf '  FAIL  %s (function %q is not defined)\n' "${desc}" "$1"
    return
  fi
  if "$@" >/dev/null 2>&1; then
    FAIL=$((FAIL + 1))
    printf '  FAIL  %s (expected non-zero exit)\n' "${desc}"
  else
    PASS=$((PASS + 1))
    printf '  ok    %s\n' "${desc}"
  fi
}

echo
echo "nvidia_lib.sh unit tests"
echo

check_eq "release: running kernel of the 2026-09-27 outage" \
  "7.0.0-1019" "$(kernel_abi_from_release 7.0.0-1019-nvidia)"
check_eq "release: the 6.17 series" \
  "6.17.0-1032" "$(kernel_abi_from_release 6.17.0-1032-nvidia)"
check_rc_nonzero "release: garbage input returns non-zero" \
  kernel_abi_from_release "not-a-kernel"

check_eq "pkgver: modules metapackage with rebuild suffix" \
  "7.0.0-1019" "$(kernel_abi_from_pkg_version '7.0.0-1019.19~24.04.2+1')"
check_eq "pkgver: image metapackage without suffix" \
  "6.17.0-1032" "$(kernel_abi_from_pkg_version '6.17.0-1032.32')"
check_rc_nonzero "pkgver: garbage input returns non-zero" \
  kernel_abi_from_pkg_version "not-a-version"

# The regression this whole task exists for: during the 2026-09-27 outage the
# running kernel was 7.0.0-1019 while the modules metapackage still targeted
# 6.17.0-1032. Those ABIs must compare unequal.
check_eq "outage: running kernel ABI differs from metapackage ABI" \
  "differ" \
  "$( [[ "$(kernel_abi_from_release 7.0.0-1019-nvidia)" \
        == "$(kernel_abi_from_pkg_version '6.17.0-1032.32')" ]] \
      && echo same || echo differ )"

# --- kernel_abi_relation ------------------------------------------------------
# The "kernel covered" row must distinguish two states that both look like an
# ABI mismatch but mean opposite things:
#   * metapkg ABI NEWER than running  = upgraded, not yet rebooted. SAFE, and
#     the ordinary state after every apt upgrade. Must NOT warn -- warning here
#     fires on every post-upgrade bring-up and re-creates the cry-wolf problem
#     this row exists to fix.
#   * metapkg ABI OLDER than running  = a pinpoint linux-image was installed
#     ahead of the metapackage pair and booted. This is the 2026-09-27 outage.
check_eq "relation: identical ABIs are covered" \
  "same" "$(kernel_abi_relation 7.0.0-1019 7.0.0-1019)"
check_eq "relation: metapkg newer than running = pending reboot (safe)" \
  "pending-reboot" "$(kernel_abi_relation 7.0.0-1019 7.0.0-1020)"
check_eq "relation: metapkg newer across series = pending reboot (safe)" \
  "pending-reboot" "$(kernel_abi_relation 6.17.0-1032 7.0.0-1019)"
check_eq "relation: metapkg OLDER than running = the 2026-09-27 exposure" \
  "exposed" "$(kernel_abi_relation 7.0.0-1019 6.17.0-1032)"
check_eq "relation: metapkg older within series = exposed" \
  "exposed" "$(kernel_abi_relation 6.17.0-1032 6.17.0-1029)"
check_rc_nonzero "relation: garbage input returns non-zero" \
  kernel_abi_relation "junk" "7.0.0-1019"

echo
echo "  ${PASS} passed, ${FAIL} failed"
echo
[[ "${FAIL}" -eq 0 ]]
