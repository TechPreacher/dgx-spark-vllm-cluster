#!/usr/bin/env bash
# Shared helpers for the NVIDIA driver check / fix / pre-reboot scripts.
# Source this file; do not execute it.
#
#   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   source "${SCRIPT_DIR}/nvidia_lib.sh"
#
# Why these scripts exist: the Sparks have NO DKMS. Kernel modules come only
# from prebuilt linux-modules-nvidia-<branch>-<kernel> packages. The kernel
# (src: linux-nvidia) and the driver (src: nvidia-graphics-drivers-<branch>)
# are separate source packages on independent *phased-update* schedules, and
# phasing is decided per-machine (deterministic on machine-id). So a single
# `apt upgrade` can pull a new kernel while holding the driver back -- on one
# node but not the other. Reboot into that gap and the running kernel has no
# nvidia.ko at all: no /dev/nvidia*, nvidia-smi dead, and every GPU container
# fails in the nvidia-container-cli prestart hook with the un-obvious
# "nvml error: driver not loaded".
#
# Seen 2026-08-20: `apt upgrade -y` took linux-image-nvidia-hwe-24.04
# 6.17.0-1026.26 -> 6.17.0-1029.29 but left
# linux-modules-nvidia-580-open-nvidia-hwe-24.04 at 6.17.0-1026.26. Reboot 14
# minutes later => no GPU on pulsar. magnetar, same command same day, was in
# the phase group and came up fine.

# Echo the name of the linux-modules-nvidia metapackage for this machine's
# driver branch (e.g. linux-modules-nvidia-580-open-nvidia-hwe-24.04).
#
# Branch is *detected*, never hardcoded, so this keeps working across a branch
# bump (580 -> 590 ...). Preference order:
#   1. the metapackage already installed (the normal case)
#   2. derived from the installed nvidia-driver-<branch> package (the case
#      where the metapackage was never installed / got autoremoved)
nvidia_modules_metapackage() {
  local pkg branch
  pkg=$(dpkg-query -W -f='${Package}\n' 'linux-modules-nvidia-*-nvidia-hwe-24.04' 2>/dev/null \
        | sed '/^$/d' | sort | head -n1) || true
  if [[ -n "${pkg}" ]]; then
    echo "${pkg}"
    return 0
  fi
  branch=$(dpkg-query -W -f='${Package}\n' 'nvidia-driver-*' 2>/dev/null \
           | sed -n 's/^nvidia-driver-\([0-9]\+\(-open\)\?\)$/\1/p' | head -n1) || true
  if [[ -n "${branch}" ]]; then
    echo "linux-modules-nvidia-${branch}-nvidia-hwe-24.04"
    return 0
  fi
  return 1
}

# Echo the kernel a reboot would land on.
#
# The Spark's grub.d drops no-grubmenu.cfg and leaves GRUB_DEFAULT=0, so GRUB
# boots its first entry = the highest-versioned installed kernel. `linux-version`
# (from linux-base) knows real Debian kernel version ordering; sort -V is the
# fallback and is right for the 6.17.0-10NN-nvidia shape we actually ship.
next_boot_kernel() {
  local k
  if command -v linux-version >/dev/null 2>&1; then
    k=$(linux-version list 2>/dev/null | linux-version sort --reverse 2>/dev/null | head -n1) || true
  fi
  if [[ -z "${k:-}" ]]; then
    k=$(find /boot -maxdepth 1 -name 'vmlinuz-*' -printf '%f\n' 2>/dev/null \
        | sed 's/^vmlinuz-//' | sort -V | tail -n1)
  fi
  [[ -n "${k}" ]] || return 1
  echo "${k}"
}

# True if a real nvidia.ko is installed for the given kernel release.
# Checks the module tree directly rather than trusting dpkg -- an interrupted
# unpack or a stray autoremove can leave the package marked installed with the
# .ko already gone.
nvidia_ko_present_for() {
  local kernel="$1"
  [[ -d "/lib/modules/${kernel}" ]] || return 1
  # NB: capture into a var rather than `| grep -q .` -- under `set -o pipefail`
  # grep -q exits on first match and SIGPIPEs find, poisoning the exit status.
  local hit
  hit=$(find "/lib/modules/${kernel}" -name 'nvidia.ko*' -print -quit 2>/dev/null)
  [[ -n "${hit}" ]]
}
