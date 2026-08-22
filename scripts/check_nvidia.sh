#!/usr/bin/env bash
set -uo pipefail
# Check the NVIDIA driver on THIS machine. Run it on each Spark separately.
#
# Catches two distinct things:
#   1. Driver broken right now -- no nvidia.ko for the running kernel, module
#      not loaded, missing /dev nodes, or a module/userspace version mismatch.
#      This is what makes `make head` die in the container prestart hook with
#      "nvml error: driver not loaded".
#   2. Driver fine now, but the metapackages have drifted out of lockstep --
#      i.e. the NEXT reboot walks into the gap. See scripts/preboot_check.sh
#      and the header of scripts/nvidia_lib.sh for why that happens.
#
# Exit codes are split so `make head` can block on (1) but not on (2):
#   0 = healthy
#   1 = driver unusable RIGHT NOW -- bring-up cannot work, fix before continuing
#   3 = driver fine now, but metapackages have drifted -- the next reboot is the
#       danger, not this run. Warn, don't block.
#   2 = script could not run (missing lib)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# A missing lib must be fatal: these scripts answer "is the GPU safe?", and a
# silently-degraded answer is worse than no answer.
if [[ ! -r "${SCRIPT_DIR}/nvidia_lib.sh" ]]; then
  echo "Cannot read ${SCRIPT_DIR}/nvidia_lib.sh -- run this script from its place in the repo." >&2
  exit 2
fi
# shellcheck source=nvidia_lib.sh
source "${SCRIPT_DIR}/nvidia_lib.sh"

FAILED=0   # driver broken now -> blocks cluster bring-up
WARNED=0   # only bites on the next reboot -> advisory
row()  { printf '  %-18s %s\n' "$1" "$2"; }
fail() { FAILED=1; row "$1" "$2"; }
warn() { WARNED=1; row "$1" "$2"; }

KERNEL=$(uname -r)
echo
echo "NVIDIA driver check -- $(hostname)"
echo
row "running kernel" "${KERNEL}"

# --- 1. is there a module at all for the kernel we are actually running? -----
if nvidia_ko_present_for "${KERNEL}"; then
  row "nvidia.ko" "PRESENT"
else
  fail "nvidia.ko" "*** MISSING for ${KERNEL} ***"
fi

# Same SIGPIPE-under-pipefail caveat as nvidia_ko_present_for: capture, don't `grep -q`.
LOADED=$(lsmod | awk '$1 == "nvidia" { print $1 }')
if [[ -n "${LOADED}" ]]; then
  row "module loaded" "YES"
else
  fail "module loaded" "*** NO ***"
fi

# The nvidia-container-toolkit prestart hook needs these device nodes; without
# them every --gpus container fails even if the module is loaded.
if [[ -e /dev/nvidiactl && -e /dev/nvidia0 ]]; then
  row "/dev/nvidia*" "PRESENT"
else
  fail "/dev/nvidia*" "*** MISSING (nvidiactl and/or nvidia0) ***"
fi

# --- 2. kernel module and userspace must be the same driver version ----------
# A mismatch is the classic "Failed to initialize NVML: Driver/library version
# mismatch" and happens when userspace upgrades but the module is still the old
# one (or vice versa).
MOD_VER=$(modinfo nvidia 2>/dev/null | awk '/^version:/{print $2}')
SMI_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1)
SMI_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)

row "kernel module ver" "${MOD_VER:-n/a}"
if [[ -n "${SMI_GPU}" ]]; then
  row "nvidia-smi" "${SMI_GPU}, ${SMI_VER}"
else
  fail "nvidia-smi" "*** FAILED: $(nvidia-smi 2>&1 | head -n1) ***"
fi

if [[ -n "${MOD_VER}" && -n "${SMI_VER}" ]]; then
  if [[ "${MOD_VER}" == "${SMI_VER}" ]]; then
    row "version match" "OK"
  else
    fail "version match" "*** MISMATCH: module ${MOD_VER} != runtime ${SMI_VER} ***"
  fi
fi

# --- 3. early warning: are the metapackages still in lockstep? ---------------
# Compare only the kernel-ABI part; the modules package carries a +N driver
# rebuild suffix (6.17.0-1029.29+1) that the image package never has.
META=$(nvidia_modules_metapackage || true)
IMG_VER=$(dpkg-query -W -f='${Version}' linux-image-nvidia-hwe-24.04 2>/dev/null || true)
MOD_PKG_VER=""
[[ -n "${META}" ]] && MOD_PKG_VER=$(dpkg-query -W -f='${Version}' "${META}" 2>/dev/null || true)

row "image metapkg" "${IMG_VER:-not installed}"
row "modules metapkg" "${MOD_PKG_VER:-not installed}${META:+  (${META})}"

if [[ -n "${IMG_VER}" && -n "${MOD_PKG_VER}" ]]; then
  if [[ "${IMG_VER}" == "${MOD_PKG_VER%%+*}" ]]; then
    row "metapkg lockstep" "OK"
  else
    warn "metapkg lockstep" "!!! DRIFTED: image ${IMG_VER} vs modules ${MOD_PKG_VER} !!!"
  fi
elif [[ -z "${MOD_PKG_VER}" ]]; then
  warn "metapkg lockstep" "!!! modules metapackage not installed !!!"
fi

# --- verdict -----------------------------------------------------------------
echo
if [[ "${FAILED}" -ne 0 ]]; then
  echo "PROBLEM  driver is not usable on $(hostname) -- the Ray cluster cannot start."
  echo "         Fix:  bash scripts/fix_nvidia.sh"
  echo
  exit 1
fi

if [[ "${WARNED}" -ne 0 ]]; then
  echo "OK (with warning)  driver works now on $(hostname) -- ${MOD_VER}"
  echo "  BUT the kernel and NVIDIA modules metapackages have drifted apart, so"
  echo "  the next reboot may come up with no GPU. Nothing is broken today."
  echo "  Before rebooting:  bash scripts/preboot_check.sh"
  echo "  To resync now:     bash scripts/fix_nvidia.sh"
  echo
  exit 3
fi

echo "OK  driver healthy on $(hostname) -- ${MOD_VER}"
echo "    Run this on the other Spark too: BOTH nodes must report the same"
echo "    driver version before bringing the Ray cluster up."
echo
exit 0
