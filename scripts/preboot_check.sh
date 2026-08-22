#!/usr/bin/env bash
set -uo pipefail
# Run this on THIS machine BEFORE rebooting it (especially after any apt
# upgrade). Run it on each Spark separately.
#
# The point: it checks the kernel the machine will boot NEXT, not the one it is
# running now. A healthy running system tells you nothing about whether the
# newly-installed kernel has NVIDIA modules -- that is exactly the gap that
# takes the GPU away on reboot. See the header of scripts/nvidia_lib.sh.
#
# Exit 0 = safe to reboot, 1 = rebooting now loses the GPU.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# A missing lib must be fatal: these scripts answer "is the GPU safe?", and a
# silently-degraded answer is worse than no answer.
if [[ ! -r "${SCRIPT_DIR}/nvidia_lib.sh" ]]; then
  echo "Cannot read ${SCRIPT_DIR}/nvidia_lib.sh -- run this script from its place in the repo." >&2
  exit 2
fi
# shellcheck source=nvidia_lib.sh
source "${SCRIPT_DIR}/nvidia_lib.sh"

row() { printf '  %-18s %s\n' "$1" "$2"; }

RUNNING=$(uname -r)
NEXT=$(next_boot_kernel || true)

echo
echo "Pre-reboot NVIDIA check -- $(hostname)"
echo
row "running kernel" "${RUNNING}"

if [[ -z "${NEXT}" ]]; then
  row "next boot kernel" "*** could not determine ***"
  echo
  echo "UNSAFE TO REBOOT  -- could not enumerate installed kernels; check /boot by hand."
  echo
  exit 1
fi

row "next boot kernel" "${NEXT}$([[ "${NEXT}" != "${RUNNING}" ]] && echo '   <-- CHANGES ON REBOOT')"

META=$(nvidia_modules_metapackage || true)
if [[ -n "${META}" ]]; then
  row "modules metapkg" "${META} $(dpkg-query -W -f='${Version}' "${META}" 2>/dev/null || echo 'NOT INSTALLED')"
else
  row "modules metapkg" "*** could not detect driver branch ***"
fi

if nvidia_ko_present_for "${NEXT}"; then
  row "nvidia.ko for next" "PRESENT"
  echo
  echo "SAFE TO REBOOT  -- ${NEXT} has NVIDIA modules."
  [[ "${NEXT}" != "${RUNNING}" ]] && \
    echo "                  (kernel changes on reboot; re-run scripts/check_nvidia.sh afterwards)"
  echo
  exit 0
fi

row "nvidia.ko for next" "*** MISSING ***"
echo
echo "UNSAFE TO REBOOT  -- ${NEXT} has NO NVIDIA modules."
echo "                    Rebooting now leaves this node with no GPU:"
echo "                    nvidia-smi dies and every --gpus container fails with"
echo "                    'nvml error: driver not loaded'."
echo
echo "  Fix first:  bash scripts/fix_nvidia.sh"
echo "  Or boot the older kernel from the GRUB menu instead."
echo
exit 1
