#!/usr/bin/env bash
set -euo pipefail
# Install the NVIDIA kernel modules matching this machine's kernel, then load
# them. Run on each Spark that scripts/check_nvidia.sh or
# scripts/preboot_check.sh flagged.
#
# Installs the *metapackage*, not the pinpoint linux-modules-nvidia-<kernel>
# package: the metapackage is what drifted out of lockstep in the first place,
# so upgrading it both fixes today and re-arms it for the next kernel.
#
# Expect apt to drag the whole NVIDIA userspace to a new driver version. That is
# correct, not collateral damage -- the modules package hard-depends on a
# matching nvidia-kernel-common-<branch>, so module and userspace move together
# by construction. It also rebuilds the PREVIOUS kernel's modules against the
# new driver, so the old kernel stays bootable as a fallback.
#
# Non-interactive: FIX_YES=1 bash scripts/fix_nvidia.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# A missing lib must be fatal: these scripts answer "is the GPU safe?", and a
# silently-degraded answer is worse than no answer.
if [[ ! -r "${SCRIPT_DIR}/nvidia_lib.sh" ]]; then
  echo "Cannot read ${SCRIPT_DIR}/nvidia_lib.sh -- run this script from its place in the repo." >&2
  exit 2
fi
# shellcheck source=nvidia_lib.sh
source "${SCRIPT_DIR}/nvidia_lib.sh"

META=$(nvidia_modules_metapackage) || {
  echo "Could not detect the NVIDIA driver branch on $(hostname)." >&2
  echo "No linux-modules-nvidia-*-nvidia-hwe-24.04 and no nvidia-driver-* installed." >&2
  exit 1
}

echo "Host:        $(hostname)"
echo "Kernel:      $(uname -r)"
echo "Metapackage: ${META}"
echo

sudo apt-get update

# Dry run first: this is where a surprise (a new linux-image being pulled in, or
# an unexpectedly large driver jump) becomes visible before it is applied.
echo "--- what this would change -------------------------------------------"
sudo apt-get install -s "${META}" | sed -n '/The following/,/^[0-9]* upgraded/p'
echo "----------------------------------------------------------------------"
echo

if [[ "${FIX_YES:-0}" != "1" ]]; then
  read -r -p "Proceed with the install above? [y/N] " reply
  [[ "${reply}" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 1; }
fi

sudo apt-get install -y "${META}"

# The new modules target the *running* kernel, so no reboot is needed -- just
# load them. nvidia_uvm pulls in nvidia and nvidia_modeset.
echo
echo "Loading modules..."
sudo modprobe nvidia_uvm

echo
exec bash "${SCRIPT_DIR}/check_nvidia.sh"
