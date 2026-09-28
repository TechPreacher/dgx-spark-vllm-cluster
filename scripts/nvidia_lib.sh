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

# --- kernel ABI extraction ---------------------------------------------------
# The 2026-09-27 outage was invisible to check_nvidia.sh's lockstep test: both
# metapackages agreed with each other (6.17.0-1032.32) while the *running*
# kernel was 7.0.0-1019, installed as a pinpoint linux-image-7.0.0-1019-nvidia
# ahead of the metapackage pair. Comparing the metapackages only to each other
# cannot see that. These two helpers extract a comparable kernel ABI from each
# side so the running kernel can be checked against the metapackage directly.
#
# Kept pure (no dpkg, no filesystem) so scripts/test_nvidia_lib.sh can test them.

# 7.0.0-1019-nvidia -> 7.0.0-1019
kernel_abi_from_release() {
  local abi
  abi=$(sed -n 's/^\([0-9]\+\.[0-9]\+\.[0-9]\+-[0-9]\+\)-.*$/\1/p' <<<"${1:-}")
  [[ -n "${abi}" ]] || return 1
  echo "${abi}"
}

# 7.0.0-1019.19~24.04.2+1 -> 7.0.0-1019   (the +N is a driver-rebuild suffix)
# 6.17.0-1032.32          -> 6.17.0-1032
kernel_abi_from_pkg_version() {
  local abi
  abi=$(sed -n 's/^\([0-9]\+\.[0-9]\+\.[0-9]\+-[0-9]\+\)\..*$/\1/p' <<<"${1:-}")
  [[ -n "${abi}" ]] || return 1
  echo "${abi}"
}

# Classify the running kernel against the kernel the modules metapackage targets.
# Echoes one of:
#   same            ABIs match -- the metapackage covers what we are running
#   pending-reboot  metapackage is NEWER: upgraded but not yet rebooted. SAFE.
#                   This is the ordinary state after any apt upgrade, so it must
#                   not be reported as drift -- doing so fires a warning on every
#                   post-upgrade bring-up and re-creates, in the other direction,
#                   the cry-wolf problem the "kernel covered" row exists to fix.
#   exposed         metapackage is OLDER: a pinpoint linux-image-<ver>-nvidia was
#                   installed ahead of the metapackage pair and booted. This is
#                   the 2026-09-27 pulsar outage, and the case this row is for.
# Returns non-zero if either ABI is unparseable.
kernel_abi_relation() {
  local run="${1:-}" meta="${2:-}"
  [[ "${run}" =~ ^[0-9]+\.[0-9]+\.[0-9]+-[0-9]+$ ]] || return 1
  [[ "${meta}" =~ ^[0-9]+\.[0-9]+\.[0-9]+-[0-9]+$ ]] || return 1
  if [[ "${run}" == "${meta}" ]]; then
    echo same
  elif dpkg --compare-versions "${meta}" gt "${run}" 2>/dev/null; then
    echo pending-reboot
  else
    echo exposed
  fi
}
