# GLM-5.3-Flash NVFP4 Rollout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve `zai-org/GLM-5.3-Flash` (320B-A18B, NVFP4 weight-only) at TP=2 across both DGX Sparks at 262,144-token context with DFlash2 speculative decoding, as a third launcher profile beside Qwen and Nemotron.

**Architecture:** Layer `ray[default]` onto the patched SM121/arm64 GHCR image using the repo's existing `cluster/Dockerfile` `BASE_IMAGE` parameter, pinned by digest. That image carries the day-0 SM121 fixes (notably the NoPE MLA attention backend, without which stock vLLM cannot run this model on GB10 at all) but ships no Ray; `cluster/Dockerfile` adds exactly that and nothing else. The Ray/RoCE bring-up, `select_up_dataplane` link enumeration and the driver preflight stay unchanged.

**Tech Stack:** Bash, Docker, Ray, vLLM, NVFP4 on SM121/GB10, RoCE over ConnectX-7.

**Spec:** `docs/superpowers/specs/2026-09-28-glm53-flash-rollout-design.md`

## Global Constraints

- Target checkpoint: `LibertAIDAI/GLM-5.3-Flash-NVFP4` (~181 GiB, weight-only NVFP4-A16).
- Base image, digest-pinned, never tag-pinned: `ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6`
- Layered image tag: `local/vllm-ray-glm53:sm121-v11-dflash2`
- `--gpu-memory-utilization` must never exceed `0.85`. `0.90` is documented to OOM on this hardware.
- `--tool-call-parser glm47` — explicitly not `glm`, not `glm45`.
- `--block-size 2304`, `--kv-cache-dtype fp8`, `--kv-cache-memory 6442450944`.
- `--enforce-eager` on ladder rungs 1–3. CUDA graph capture is a separate, later experiment.
- `PROFILE` is mandatory on every `make` bring-up target and has no default.
- Both nodes must report the identical NVIDIA driver version before any Ray bring-up. Currently satisfied: both on 580.178.04.
- Every image build runs on **both** nodes — local tags are not registry-backed.
- Do **not** set `NCCL_IB_HCA` / `NCCL_SOCKET_IFNAME` / `UCX_NET_DEVICES` in any profile. `select_up_dataplane` builds those from carrier-up links at launch; a static value hands NCCL a down HCA after a warm reboot.
- DFlash2 drafter weights are CC-BY-NC-ND-4.0: research/personal use, never redistributed, never baked into a shared image.
- One model at a time. UMA cannot hold GLM and Nemotron concurrently.

## Review Focus

Five failure modes the spec implies that no task's happy path exercises. Each has a test added to the task that owns the code.

1. **Bogus profile name** — `make head PROFILE=typo` must fail naming valid profiles, not source nothing and bring up a cluster with no forwarded env. (Task 5)
2. **Profile/image mismatch** — cluster brought up on the Nemotron image, then `glm/launch-glm53-flash.sh` run against it. Must refuse, not fail deep inside vLLM on an unknown architecture. (Task 4)
3. **Warm-reboot link degradation** — after a warm reboot only 2 of 4 CX7 halves have carrier. The GLM profile must not regress this; bring-up must still succeed on 2 links. (Task 3)
4. **Missing credentials** — `glm/.env` absent or `HF_TOKEN` unset must fail immediately with a named cause, not 40 minutes into a 181 GiB download. (Task 4)
5. **Digest drift** — the layered image must be provably built from the pinned digest, so a moved upstream tag cannot silently substitute different kernels. (Task 2)

---

### Task 1: Close the driver-detector blind spot

The 2026-09-27 outage was invisible to `check_nvidia.sh`: it printed `metapkg lockstep OK` throughout, because both metapackages agreed with each other at 6.17.0-1032.32. The drift was that a **pinpoint** `linux-image-7.0.0-1019-nvidia` was installed ahead of the metapackage pair and then booted. The lockstep check compares the metapackages to each other but never to the running kernel.

**Files:**
- Modify: `scripts/nvidia_lib.sh` (append two helpers)
- Modify: `scripts/check_nvidia.sh` (extend section 3)
- Test: `scripts/test_nvidia_lib.sh` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `kernel_abi_from_release <release>` → echoes ABI (`7.0.0-1019-nvidia` → `7.0.0-1019`), returns 1 if unparseable. `kernel_abi_from_pkg_version <version>` → echoes ABI (`7.0.0-1019.19~24.04.2+1` → `7.0.0-1019`), returns 1 if unparseable.

- [ ] **Step 1: Write the failing test**

Create `scripts/test_nvidia_lib.sh`:

```bash
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

check_rc_nonzero() {
  local desc="$1"; shift
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

echo
echo "  ${PASS} passed, ${FAIL} failed"
echo
[[ "${FAIL}" -eq 0 ]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `bash scripts/test_nvidia_lib.sh`
Expected: FAIL — every `kernel_abi_*` line errors with `command not found`, and the summary reports failures with a non-zero exit.

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/nvidia_lib.sh`:

```bash
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `bash scripts/test_nvidia_lib.sh`
Expected: PASS — `7 passed, 0 failed`, exit 0.

- [ ] **Step 5: Wire the check into check_nvidia.sh**

In `scripts/check_nvidia.sh`, immediately after the existing `metapkg lockstep` if/elif block (the end of section 3) and before the `# --- verdict ---` banner, insert:

```bash
# --- 3b. does the modules metapackage cover the kernel we are RUNNING? -------
# Section 3 compares the two metapackages to each other, which cannot see a
# pinpoint linux-image-<ver>-nvidia installed ahead of the metapackage pair and
# then booted. That is exactly the 2026-09-27 pulsar outage: both metapackages
# agreed at 6.17.0-1032.32 while the running kernel was 7.0.0-1019 with no
# nvidia.ko, and this script still printed "metapkg lockstep OK".
RUN_ABI=$(kernel_abi_from_release "${KERNEL}" 2>/dev/null || true)
META_ABI=""
[[ -n "${MOD_PKG_VER}" ]] && META_ABI=$(kernel_abi_from_pkg_version "${MOD_PKG_VER}" 2>/dev/null || true)

if [[ -n "${RUN_ABI}" && -n "${META_ABI}" ]]; then
  if [[ "${RUN_ABI}" == "${META_ABI}" ]]; then
    row "metapkg covers kernel" "OK (${RUN_ABI})"
  else
    warn "metapkg covers kernel" "!!! running ${RUN_ABI} but modules metapkg targets ${META_ABI} !!!"
  fi
fi
```

Note it uses `warn`, not `fail`: when a pinpoint modules package happens to cover the running kernel the GPU works fine today, and the real exposure is the next upgrade. When it is *not* covered, section 1's `nvidia.ko MISSING` has already set `FAILED`, and this row now supplies the cause the operator was previously missing.

- [ ] **Step 6: Verify the new row appears on a healthy node**

Run: `bash scripts/check_nvidia.sh; echo "exit=$?"`
Expected: output includes `metapkg covers kernel   OK (7.0.0-1019)` and `exit=0`.

- [ ] **Step 7: Verify the check would have caught the outage**

Run:
```bash
bash -c 'source scripts/nvidia_lib.sh
  r=$(kernel_abi_from_release 7.0.0-1019-nvidia)
  m=$(kernel_abi_from_pkg_version 6.17.0-1032.32)
  [[ "$r" != "$m" ]] && echo "WOULD HAVE WARNED: running $r, metapkg $m"'
```
Expected: `WOULD HAVE WARNED: running 7.0.0-1019, metapkg 6.17.0-1032`

- [ ] **Step 8: Commit**

```bash
git add scripts/nvidia_lib.sh scripts/check_nvidia.sh scripts/test_nvidia_lib.sh
git commit -m "Detect a pinpoint kernel installed ahead of the metapackage pair

check_nvidia.sh compared the two metapackages only to each other, so the
2026-09-27 pulsar outage read as 'metapkg lockstep OK' while the running
kernel had no nvidia.ko: a pinpoint linux-image-7.0.0-1019-nvidia had been
installed ahead of the pair and booted.

Add two pure ABI-extraction helpers plus unit tests for them, and compare the
running kernel against the modules metapackage directly. Warns rather than
fails: when a pinpoint modules package covers the running kernel the GPU is
fine today and the exposure is the next upgrade; when it does not, nvidia.ko
is already reported missing and this row supplies the cause."
```

---

### Task 2: Build and verify the Ray-layered GLM image

The GHCR image ships "only vLLM + our patches" — no Ray. `cluster/Dockerfile` already does exactly one thing (`pip install "ray[default]"`) on a parameterised `BASE_IMAGE`, so no new Dockerfile is needed. The risk this introduces is that `ray[default]`'s dependency resolution perturbs the pins that make SM121 correct — FlashInfer 0.6.18 in particular, whose 0.6.17 predecessor produced NaN at batch 64–256 rows. That failure would appear as silently wrong numerics, not a build error, so it gets a gate.

**Files:**
- Create: `glm/verify-image.sh`
- Test: `glm/verify-image.sh` is itself the test.

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: image tag `local/vllm-ray-glm53:sm121-v11-dflash2` present on both nodes. `glm/verify-image.sh` exits 0 when the layered image's Python environment differs from the base only by additions.

- [ ] **Step 1: Write the verification script**

Create `glm/verify-image.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
# Verify the Ray-layered GLM image against the base image it was built from.
#
# Why this exists: the base image's pins are load-bearing for SM121 correctness.
# It exists because stock vLLM cannot run GLM-5.3-Flash on GB10 at all (the
# sparse-attention kernel assumes DeepSeek's pe_dim=64; this model has
# qk_rope_head_dim=0), and the patched build also pins FlashInfer 0.6.18
# specifically because 0.6.17 produced NaN on SM121 at batch 64-256 rows, plus
# NCCL 2.30.7 and CUTLASS 4.6.2 after a nightly silently skewed them.
#
# If `pip install ray[default]` moves any of those, we get wrong numbers rather
# than an error. So: the layered image may only ADD packages to the base. Any
# version change or removal is a hard failure.
#
# Run on each node after building. Usage: bash glm/verify-image.sh

BASE_DIGEST="ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6"
LAYERED_TAG="${LAYERED_TAG:-local/vllm-ray-glm53:sm121-v11-dflash2}"

freeze() {
  docker run --rm --entrypoint /bin/bash "$1" -c \
    'pip freeze --disable-pip-version-check 2>/dev/null | sed "/^-e /d" | sort'
}

echo "Base:    ${BASE_DIGEST}"
echo "Layered: ${LAYERED_TAG}"
echo

if ! docker image inspect "${LAYERED_TAG}" >/dev/null 2>&1; then
  echo "FAIL: ${LAYERED_TAG} does not exist on $(hostname). Build it first:" >&2
  echo "  BASE_IMAGE=${BASE_DIGEST} TAG=${LAYERED_TAG} bash cluster/build-image.sh" >&2
  exit 1
fi

# Provenance: the layered image must descend from the pinned digest, so a moved
# upstream tag cannot silently substitute different kernels underneath us.
BUILT_FROM=$(docker image inspect "${LAYERED_TAG}" \
  --format '{{index .Config.Labels "glm.base.digest"}}' 2>/dev/null || true)
if [[ "${BUILT_FROM}" != "${BASE_DIGEST}" ]]; then
  echo "FAIL: ${LAYERED_TAG} does not record the pinned base digest." >&2
  echo "  recorded: ${BUILT_FROM:-<none>}" >&2
  echo "  expected: ${BASE_DIGEST}" >&2
  echo "Rebuild with cluster/build-image.sh, which stamps the label." >&2
  exit 1
fi
echo "provenance  OK (built from pinned digest)"

TMP=$(mktemp -d)
trap 'rm -rf "${TMP}"' EXIT
freeze "${BASE_DIGEST}"  > "${TMP}/base.txt"
freeze "${LAYERED_TAG}"  > "${TMP}/layered.txt"

# A changed version shows up as a removal of the base line; a pure addition does
# not. So: every line present in base must still be present, byte-identical.
CHANGED=$(comm -23 "${TMP}/base.txt" "${TMP}/layered.txt")
if [[ -n "${CHANGED}" ]]; then
  echo >&2
  echo "FAIL: the Ray layer changed or removed packages the base image pinned:" >&2
  while read -r line; do
    [[ -z "${line}" ]] && continue
    name="${line%%==*}"
    now=$(grep -i "^${name}==" "${TMP}/layered.txt" || echo "<removed>")
    printf '  %-40s base: %-24s layered: %s\n' "${name}" "${line#*==}" "${now#*==}" >&2
  done <<<"${CHANGED}"
  echo >&2
  echo "These pins are why SM121 works. Pin ray to a version that leaves them" >&2
  echo "alone, or fall back to approach B (local rebuild of the v1->v11 chain)." >&2
  exit 1
fi
echo "pins        OK (no base package changed or removed)"

ADDED=$(comm -13 "${TMP}/base.txt" "${TMP}/layered.txt" | wc -l)
echo "additions   ${ADDED} package(s) added by the ray layer"

if ! docker run --rm --entrypoint /bin/bash "${LAYERED_TAG}" -c 'ray --version' >/dev/null 2>&1; then
  echo "FAIL: ray CLI is not callable in ${LAYERED_TAG}." >&2
  exit 1
fi
echo "ray         OK ($(docker run --rm --entrypoint /bin/bash "${LAYERED_TAG}" -c 'ray --version' 2>&1 | head -n1))"

echo
echo "OK  ${LAYERED_TAG} is safe to use on $(hostname)."
```

- [ ] **Step 2: Run it to verify it fails**

Run: `bash glm/verify-image.sh; echo "exit=$?"`
Expected: FAIL with `FAIL: local/vllm-ray-glm53:sm121-v11-dflash2 does not exist`, `exit=1`.

- [ ] **Step 3: Stamp the base digest in cluster/build-image.sh**

`verify-image.sh` checks a `glm.base.digest` label that nothing writes yet. Modify `cluster/build-image.sh` so the build records what it was built from — add `--label` to the existing `docker build` invocation:

```bash
docker build \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  --label "glm.base.digest=${BASE_IMAGE}" \
  -t "${TAG}" \
  "${SCRIPT_DIR}"
```

This is harmless for the existing Nemotron/Qwen image (it just records the NGC base) and is what makes digest provenance checkable.

- [ ] **Step 4: Build the layered image on pulsar**

Run:
```bash
BASE_IMAGE=ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6 \
TAG=local/vllm-ray-glm53:sm121-v11-dflash2 \
bash cluster/build-image.sh
```
Expected: the pull fetches by digest, the `pip install ray[default]` layer succeeds, and `ray --version` prints during build.

Note: this pulls a large image. If disk is tight, `docker image prune` the dangling layers from previous builds first — do **not** remove `local/vllm-ray:26.05.post1`, which the Nemotron and Qwen paths still need.

- [ ] **Step 5: Run the verification**

Run: `bash glm/verify-image.sh; echo "exit=$?"`
Expected: `provenance OK`, `pins OK`, `ray OK`, `exit=0`.

If it reports changed pins, stop. Do not proceed to Task 3. Try pinning Ray (edit `cluster/Dockerfile`'s `pip install` to a specific `ray[default]==X.Y.Z` and rebuild); if no version leaves the pins intact, the spec's fallback is approach B, a local rebuild of the `v1→v8→v11-dflash2` chain.

- [ ] **Step 6: Repeat on magnetar**

Run on `magnetar` (local tags are not registry-backed, so each node builds its own):
```bash
ssh magnetar 'cd ~/vllm-cluster && git fetch && git checkout glm53-flash-rollout && git pull'
ssh magnetar 'cd ~/vllm-cluster && BASE_IMAGE=ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6 TAG=local/vllm-ray-glm53:sm121-v11-dflash2 bash cluster/build-image.sh'
ssh magnetar 'cd ~/vllm-cluster && bash glm/verify-image.sh'
```
Expected: `exit=0` on magnetar too.

- [ ] **Step 7: Commit**

```bash
git add glm/verify-image.sh cluster/build-image.sh
git commit -m "Add GLM image verification and stamp base digest on builds

The patched SM121 image ships vLLM plus its patches but no Ray, so the GLM
image is cluster/Dockerfile's existing ray[default] layer over a different
BASE_IMAGE. The risk is that ray's resolver moves the pins that make SM121
correct -- FlashInfer 0.6.18 above all, whose predecessor produced NaN at
batch 64-256 rows on this hardware, which would surface as wrong numbers
rather than a build failure.

verify-image.sh therefore allows the layer to ADD packages and fails on any
version change or removal, and checks the image descends from the pinned
digest so a moved upstream tag cannot substitute different kernels.
build-image.sh now stamps that digest as a label."
```

---

### Task 3: Discover the real runtime env and drafter identity

The spec deliberately left two values undetermined rather than guessing them from prose recipes: the `VLLM_FORWARD_VARS` set and the DFlash2 drafter's `--speculative-config` spelling. Both are read off the image. This matters because a rank-1 env mismatch across Ray nodes manifests as a collective hang, not an error — the Nemotron path needed four such vars for exactly this reason.

**Files:**
- Create: `glm/cluster-env.sh`
- Create: `glm/DISCOVERY.md` (records what was found and where, so the next person does not re-derive it)

**Interfaces:**
- Consumes: `local/vllm-ray-glm53:sm121-v11-dflash2` from Task 2.
- Produces: `glm/cluster-env.sh` exporting the GLM runtime vars plus `VLLM_FORWARD_VARS` naming them; `GLM_SPEC_CONFIG` recorded in `glm/DISCOVERY.md` for Task 7.

- [ ] **Step 1: Inspect the image's own launcher and baked env**

Run:
```bash
IMG=local/vllm-ray-glm53:sm121-v11-dflash2
docker image inspect "$IMG" --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -Ei 'vllm|flashinfer|nccl|torch|cuda' | sort
docker run --rm --entrypoint /bin/bash "$IMG" -c 'ls -la /opt /workspace 2>/dev/null; find / -maxdepth 3 -name "launch*glm*" -o -maxdepth 3 -name "*dflash*" 2>/dev/null | head -20'
```
Record every `VLLM_*` variable the image bakes in, and locate any launcher script it ships.

- [ ] **Step 2: Read the shipped launcher for the serve flags**

Run (substituting the path found in Step 1):
```bash
docker run --rm --entrypoint /bin/bash "$IMG" -c 'cat <path-to-launcher>'
```
Extract verbatim: the `--speculative-config` (or `--speculative-model`) spelling and its JSON, the drafter's HuggingFace repo id, and every `export VLLM_*` the launcher performs before serving.

The distinction that matters: a var the **image** already bakes into `Config.Env` is inherited by every container from that image on both nodes and does **not** need forwarding. A var the **launcher** exports at runtime is set only in the head's exec context, is invisible to Ray-spawned rank 1 on the worker, and **must** go in `VLLM_FORWARD_VARS`.

- [ ] **Step 3: Confirm the drafter is fetchable**

Run:
```bash
source glm/.env 2>/dev/null || true
curl -fsSL -H "Authorization: Bearer ${HF_TOKEN}" \
  "https://huggingface.co/api/models/<drafter-repo-id>" | head -c 400; echo
```
Expected: JSON metadata, not a 401/404. Note the licence field — expect CC-BY-NC-ND-4.0, which is why this path is research-only.

- [ ] **Step 4: Write glm/cluster-env.sh**

Create `glm/cluster-env.sh`, substituting the vars discovered in Step 2 for the placeholder block. The structure, comments and `VLLM_FORWARD_VARS` mechanism are fixed; only the variable list is discovery-driven:

```bash
# Source this file BEFORE bringing up the Ray cluster when you intend to run
# LibertAIDAI/GLM-5.3-Flash-NVFP4. These vars must be present in the Ray
# container env at START time on BOTH nodes -- Ray cannot propagate them from
# the head driver across nodes to worker ranks at vllm-serve time, so a var set
# only on the head gives rank 1 a different backend and the run hangs in a
# collective rather than failing with a message.
#
# Usage:
#   # On Node 1 (head):
#   source glm/cluster-env.sh
#   make head PROFILE=glm
#
#   # On Node 2 (worker):
#   source glm/cluster-env.sh
#   make worker PROFILE=glm
#
#   # Then on Node 1 in a new terminal:
#   make serve PROFILE=glm
#
# Source order matters: run_*node_2.sh expands VLLM_FORWARD_VARS at script
# start, so these must already be exported in the parent shell.
#
# Deliberately NOT set here: NCCL_IB_HCA, NCCL_SOCKET_IFNAME, UCX_NET_DEVICES.
# select_up_dataplane (cluster/lib.sh) builds those at launch from the CX7 links
# that currently have carrier -- all 4 after a cold boot, only the 2 f0 halves
# after a warm reboot. A static value here would hand NCCL a down HCA after a
# warm reboot and stall collective init.

export VLLM_IMAGE=local/vllm-ray-glm53:sm121-v11-dflash2

# <-- Replace this block with the vars discovered in Task 3 Step 2. -->
# <-- Every name added here must also be appended to VLLM_FORWARD_VARS. -->
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

export VLLM_FORWARD_VARS="VLLM_ALLOW_LONG_MAX_MODEL_LEN"
```

Note `VLLM_IMAGE` is exported here rather than passed on the command line: `run_headnode_2.sh` already honours `VLLM_IMAGE` with a default, so the profile owning its own image is what makes `make head PROFILE=glm` bring up the right container.

- [ ] **Step 5: Verify the vars survive into a container**

Run:
```bash
source glm/cluster-env.sh
docker run --rm $(for V in ${VLLM_FORWARD_VARS}; do printf -- '-e %s=%s ' "$V" "${!V}"; done) \
  --entrypoint /bin/bash "${VLLM_IMAGE}" -c \
  'for V in '"${VLLM_FORWARD_VARS}"'; do printf "%-40s %s\n" "$V" "${!V:-<UNSET>}"; done'
```
Expected: every variable printed with its value, none `<UNSET>`.

- [ ] **Step 6: Verify the warm-reboot link path is unaffected (Review Focus 3)**

The GLM profile must not regress the warm-reboot handling. Confirm it sets no data-plane var:

```bash
source glm/cluster-env.sh
for V in NCCL_IB_HCA NCCL_SOCKET_IFNAME UCX_NET_DEVICES GLOO_SOCKET_IFNAME; do
  printf '%-24s %s\n' "$V" "${!V:-<unset, correct>}"
done
grep -c 'NCCL_IB_HCA\|NCCL_SOCKET_IFNAME\|UCX_NET_DEVICES' glm/cluster-env.sh
```
Expected: all four report `<unset, correct>`, and the grep count is `0` outside the explanatory comment — adjust the grep to exclude comment lines if needed. Then confirm live enumeration still works:

```bash
bash -c 'source cluster/lib.sh && select_up_dataplane && echo "DATA_IFS=$DATA_IFS"'
```
Expected: `ConnectX-7 data-plane links up: N/4` with N ≥ 2, and a non-empty `DATA_IFS`.

- [ ] **Step 7: Record the discovery**

Create `glm/DISCOVERY.md` capturing, with the command that produced each: the image digest inspected, every `VLLM_*` baked into `Config.Env`, every var the shipped launcher exports (and which of those went into `VLLM_FORWARD_VARS` and why), the drafter repo id and its licence, and the exact `--speculative-config` JSON for Task 7. This is the artifact that stops the next person re-deriving it from blog posts.

- [ ] **Step 8: Commit**

```bash
git add glm/cluster-env.sh glm/DISCOVERY.md
git commit -m "Add glm profile env, discovered from the image not from prose

Ray does not propagate the head driver's environment to worker ranks across
nodes, so a var set only on the head gives rank 1 a different backend and the
run hangs in a collective instead of failing. The Nemotron path needed four
such vars; this records GLM's, read off the image's own launcher rather than
guessed from recipes, along with the DFlash2 drafter id and spec-config JSON.

Sets no NCCL/UCX data-plane var on purpose: select_up_dataplane builds those
from carrier-up links, which is what keeps bring-up working after a warm
reboot sheds the f1 half of each CX7 port."
```

---

### Task 4: The GLM launcher

**Files:**
- Create: `glm/launch-glm53-flash.sh`
- Create: `glm/.env.example`
- Create: `glm/README.md`

**Interfaces:**
- Consumes: `glm/cluster-env.sh` (Task 3), the layered image (Task 2), `find_ray_container` and `load_env` from `cluster/lib.sh`.
- Produces: `glm/launch-glm53-flash.sh`, honouring `MAX_MODEL_LEN`, `GPU_MEM_UTIL`, `KV_CACHE_MEMORY`, `ENABLE_EAGER`, `ENABLE_DFLASH2`, `REASONING_PARSER`, `PORT`.

- [ ] **Step 1: Write the launcher**

Create `glm/launch-glm53-flash.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

# Launch LibertAIDAI/GLM-5.3-Flash-NVFP4 on the running Ray-clustered vLLM
# container on Node 1. Ray dispatches shard 2 to Node 2 over the RoCE data
# plane established by run_headnode_2.sh / run_workernode_2.sh.
#
# Does NOT docker run: the Ray cluster must already be up, brought up with
# glm/cluster-env.sh sourced on BOTH nodes.
#
# Model: 320B total / 18B active MoE, natively multimodal, hybrid sparse +
# linear attention with Manifold-Constrained Hyper-Connections. MIT.
# Quantization: weight-only NVFP4-A16 -- the routed-expert FFN tensors (97% of
# parameters) are NVFP4 (E2M1, FP8-E4M3 per-16-block scales); both attention
# flavours, the vision tower, shared experts, routers, embeddings and the LM
# head stay BF16.
#
# ---------------------------------------------------------------------------
# Memory: tighter than Nemotron, which is why the defaults are what they are
# ---------------------------------------------------------------------------
#   weights        181 GiB    -> 90.5 GiB / node at TP=2
#   budget @ 0.85  0.85 x 121.63 GiB = 103.4 GiB / node
#   headroom       ~12.9 GiB / node for KV + activations + graphs
#
# Nemotron runs 1M context with roughly twice this headroom. Consequences:
#   * GPU_MEM_UTIL 0.85 is a CEILING, not a starting point. 0.90 is documented
#     to OOM on this hardware.
#   * KV is fp8 with an explicit 6 GiB budget rather than "whatever is left".
#   * ENABLE_EAGER defaults ON. CUDA graph capture is a memory spike, and
#     capture_end is historically where the cgroup-permission failure first
#     surfaced. Turn it off only after 262K is proven stable.
#
# Host-stability context: a gpt-oss-120b run once starved this host until sshd
# was unreachable while ICMP still replied, and recovery needed a power cycle.
# run_cluster.sh sets no --memory cgroup cap, so this launcher cannot add one.
# Required hardening on BOTH nodes before running this:
#   sudo systemctl edit ssh        # [Service] / OOMScoreAdjust=-1000
#   sudo apt install earlyoom && sudo systemctl enable --now earlyoom
# ---------------------------------------------------------------------------
#
# LICENCE: the DFlash2 drafter is CC-BY-NC-ND-4.0 -- research / personal use
# only. Do not redistribute it and do not bake it into a shared image. The
# target model itself is MIT. Set ENABLE_DFLASH2=0 for a licence-clean run
# (slower: the published figures are 46.9 tok/s with DFlash2 vs 21.8 with MTP).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../cluster/lib.sh
source "${SCRIPT_DIR}/../cluster/lib.sh"
load_env "${SCRIPT_DIR}"
: "${VLLM_API_KEY:?VLLM_API_KEY not set (expected in glm/.env -- copy glm/.env.example)}"
: "${HF_TOKEN:?HF_TOKEN not set (expected in glm/.env -- copy glm/.env.example)}"

# --- Overridable knobs -------------------------------------------------------
MODEL_CKPT="${MODEL_CKPT:-LibertAIDAI/GLM-5.3-Flash-NVFP4}"
SERVED_NAME="${SERVED_NAME:-zai-org/glm-5.3-flash}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
KV_CACHE_MEMORY="${KV_CACHE_MEMORY:-6442450944}"
BLOCK_SIZE="${BLOCK_SIZE:-2304}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
TP_SIZE="${TP_SIZE:-2}"
PORT="${PORT:-8000}"
ENABLE_EAGER="${ENABLE_EAGER:-1}"
ENABLE_DFLASH2="${ENABLE_DFLASH2:-0}"
# Recipes disagree: the checkpoint card says deepseek_r1, one 2-Spark recipe
# says glm45. A wrong parser does not error -- it silently mis-splits
# reasoning_content from content -- so this is probed in Task 6, not assumed.
REASONING_PARSER="${REASONING_PARSER:-deepseek_r1}"
EXPECTED_IMAGE="${EXPECTED_IMAGE:-local/vllm-ray-glm53:sm121-v11-dflash2}"

# Refuse to exceed the documented OOM ceiling, however the caller was invoked.
if awk "BEGIN{exit !(${GPU_MEM_UTIL} > 0.85)}"; then
  echo "ERROR: GPU_MEM_UTIL=${GPU_MEM_UTIL} exceeds 0.85." >&2
  echo "0.90 is documented to OOM on GB10 with this checkpoint. Refusing." >&2
  exit 1
fi

VLLM_CONTAINER=$(find_ray_container)

# The cluster must be running the GLM image. Bringing it up on the Nemotron
# image and then exec'ing this in fails deep inside vLLM on an unknown
# architecture; catch it here with a cause instead.
RUNNING_IMAGE=$(docker inspect --format '{{.Config.Image}}' "${VLLM_CONTAINER}")
if [[ "${RUNNING_IMAGE}" != "${EXPECTED_IMAGE}" ]]; then
  cat >&2 <<EOF
ERROR: container ${VLLM_CONTAINER} is running the wrong image.
  running:  ${RUNNING_IMAGE}
  expected: ${EXPECTED_IMAGE}

The Ray cluster was brought up on a different profile. Tear it down and bring
it back up with the glm profile on BOTH nodes:

  source glm/cluster-env.sh && make head   PROFILE=glm    # Node 1
  source glm/cluster-env.sh && make worker PROFILE=glm    # Node 2
EOF
  exit 1
fi

echo "Using container: ${VLLM_CONTAINER}  (${RUNNING_IMAGE})"
echo "  model:             ${MODEL_CKPT}"
echo "  TP:                ${TP_SIZE}"
echo "  max-model-len:     ${MAX_MODEL_LEN}"
echo "  gpu-mem-util:      ${GPU_MEM_UTIL}"
echo "  kv-cache-memory:   ${KV_CACHE_MEMORY}"
echo "  block-size:        ${BLOCK_SIZE}"
echo "  enforce-eager:     ${ENABLE_EAGER}"
echo "  DFlash2 spec:      ${ENABLE_DFLASH2}"
echo "  reasoning parser:  ${REASONING_PARSER}"
echo "  port:              ${PORT}"

# Same guard as the Nemotron launcher: if the forwarded vars are absent from the
# head container's env, the user did not source glm/cluster-env.sh before
# bring-up. Rank 1 on the worker will not have them either, and the run hangs in
# a collective rather than erroring. Fail here with the fix instead.
FORWARD_VARS=$(bash -c 'source '"${SCRIPT_DIR}"'/cluster-env.sh >/dev/null 2>&1; echo "${VLLM_FORWARD_VARS}"')
MISSING_VARS=$(docker exec "${VLLM_CONTAINER}" /bin/bash -c '
  set -u
  missing=""
  for V in '"${FORWARD_VARS}"'; do
    [[ -z "${!V:-}" ]] && missing="${missing} $V"
  done
  echo "${missing}"
' | xargs)
if [[ -n "${MISSING_VARS}" ]]; then
  cat >&2 <<EOF
ERROR: Required GLM env vars are not set inside the Ray container:
  ${MISSING_VARS}

These must be present at container START time on BOTH nodes; they cannot be
added now via docker exec, because Ray-spawned rank-1 workers on the worker
node would still be missing them. Tear the cluster down and bring it back up:

  source glm/cluster-env.sh && make head   PROFILE=glm    # Node 1
  source glm/cluster-env.sh && make worker PROFILE=glm    # Node 2
EOF
  exit 1
fi

# Do not serve until Ray actually reports both nodes. The 2-Spark recipes sleep
# ~25s here; polling the real condition is strictly better than a magic number.
# Count GPUs rather than parsing the node list: `ray status` prints a
# "Resources" block with a "0.0/2.0 GPU" usage line, and each Spark contributes
# exactly one GB10. That line is far more stable across Ray versions than the
# "Active:" node listing, whose formatting has changed between releases.
ray_total_gpus() {
  docker exec "${VLLM_CONTAINER}" /bin/bash -c \
    "ray status 2>/dev/null | sed -n 's#.*/\([0-9.]*\) GPU\$#\1#p' | head -n1" 2>/dev/null \
    | cut -d. -f1
}

echo -n "Waiting for Ray to report 2 GPUs"
ALIVE=0
for _ in $(seq 1 60); do
  ALIVE=$(ray_total_gpus)
  ALIVE="${ALIVE:-0}"
  [[ "${ALIVE}" -ge 2 ]] && { echo " -- ${ALIVE} GPUs"; break; }
  echo -n "."
  sleep 2
done
if [[ "${ALIVE}" -lt 2 ]]; then
  echo
  echo "ERROR: Ray reports ${ALIVE} GPU(s), expected 2." >&2
  echo "The worker has not joined. On Node 2:" >&2
  echo "  source glm/cluster-env.sh && make worker PROFILE=glm" >&2
  exit 1
fi

EAGER_FLAG=""
[[ "${ENABLE_EAGER}" == "1" ]] && EAGER_FLAG="--enforce-eager"

# Filled in from glm/DISCOVERY.md (Task 3). Left empty until DFlash2 is enabled
# at ladder rung 4.
SPEC_FLAG=""
if [[ "${ENABLE_DFLASH2}" == "1" ]]; then
  SPEC_FLAG="${GLM_SPEC_CONFIG:?GLM_SPEC_CONFIG not set -- see glm/DISCOVERY.md for the exact --speculative-config JSON}"
fi

docker exec -it \
  -e VLLM_API_KEY="${VLLM_API_KEY}" \
  -e HF_TOKEN="${HF_TOKEN}" \
  -e MODEL_CKPT="${MODEL_CKPT}" \
  -e SERVED_NAME="${SERVED_NAME}" \
  -e MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
  -e GPU_MEM_UTIL="${GPU_MEM_UTIL}" \
  -e KV_CACHE_MEMORY="${KV_CACHE_MEMORY}" \
  -e BLOCK_SIZE="${BLOCK_SIZE}" \
  -e MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
  -e TP_SIZE="${TP_SIZE}" \
  -e PORT="${PORT}" \
  -e EAGER_FLAG="${EAGER_FLAG}" \
  -e SPEC_FLAG="${SPEC_FLAG}" \
  -e REASONING_PARSER="${REASONING_PARSER}" \
  "${VLLM_CONTAINER}" /bin/bash -c '
    set -euo pipefail
    # shellcheck disable=SC2086
    exec vllm serve "${MODEL_CKPT}" \
      --served-model-name "${SERVED_NAME}" \
      --host 0.0.0.0 \
      --port "${PORT}" \
      --tensor-parallel-size "${TP_SIZE}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --gpu-memory-utilization "${GPU_MEM_UTIL}" \
      --kv-cache-dtype fp8 \
      --kv-cache-memory "${KV_CACHE_MEMORY}" \
      --block-size "${BLOCK_SIZE}" \
      --max-num-seqs "${MAX_NUM_SEQS}" \
      --enable-auto-tool-choice \
      --tool-call-parser glm47 \
      --reasoning-parser "${REASONING_PARSER}" \
      --skip-mm-profiling \
      ${EAGER_FLAG} \
      ${SPEC_FLAG}
  '
```

Note `--tool-call-parser glm47` is hardcoded rather than env-driven: both sources agree on it and explicitly warn against `glm` and `glm45`, so it should not be casually overridable.

- [ ] **Step 2: Write glm/.env.example**

```bash
# Copy to glm/.env and fill in. glm/.env is gitignored by the repo's **/.env rule.
#
# HF_TOKEN     -- needs accepted terms for LibertAIDAI/GLM-5.3-Flash-NVFP4 and,
#                 if using DFlash2, for the drafter repo (CC-BY-NC-ND-4.0).
# VLLM_API_KEY -- the bearer token clients present to :8000.
HF_TOKEN=
VLLM_API_KEY=
```

- [ ] **Step 3: Verify the credential guard fires (Review Focus 4)**

Run with no `.env` present:
```bash
mv glm/.env glm/.env.bak 2>/dev/null || true
bash glm/launch-glm53-flash.sh; echo "exit=$?"
mv glm/.env.bak glm/.env 2>/dev/null || true
```
Expected: fails immediately with `VLLM_API_KEY not set (expected in glm/.env -- copy glm/.env.example)` and a non-zero exit — before any container lookup or model download.

- [ ] **Step 4: Verify the util ceiling guard fires**

Run: `GPU_MEM_UTIL=0.90 bash glm/launch-glm53-flash.sh; echo "exit=$?"`
Expected: `ERROR: GPU_MEM_UTIL=0.90 exceeds 0.85.`, `exit=1`.

- [ ] **Step 5: Verify the wrong-image guard fires (Review Focus 2)**

With the cluster up on the **Nemotron** profile:
```bash
source nemotron/cluster-env.sh && make head PROFILE=nemotron   # Node 1, in its own terminal
bash glm/launch-glm53-flash.sh; echo "exit=$?"
```
Expected: `ERROR: container node-... is running the wrong image.` naming both images, `exit=1`. Tear the cluster down afterwards.

- [ ] **Step 6: Verify .env is not tracked**

Run:
```bash
git check-ignore -v glm/.env
git check-ignore -v glm/.env.example || echo "glm/.env.example not ignored (correct)"
```
Expected: `.gitignore:3:**/.env	glm/.env` for the first, and `not ignored (correct)` for the second — the rule matches a file *named* `.env`, so the example file is still committable. Verified on this repo 2026-09-28.

- [ ] **Step 7: Write glm/README.md**

Cover: the memory budget and why 0.85 is a ceiling; the full bring-up sequence with `PROFILE=glm`; the knobs and their defaults; the DFlash2 licence constraint stated plainly; the one-model-at-a-time invariant; and a pointer to `glm/DISCOVERY.md` for the forwarded-var provenance.

- [ ] **Step 8: Commit**

```bash
git add glm/launch-glm53-flash.sh glm/.env.example glm/README.md
git commit -m "Add the GLM-5.3-Flash launcher

Follows the Nemotron launcher's shape -- docker exec into the running head
container, no docker run of its own -- with three guards it does not have:
a hard refusal above gpu-memory-utilization 0.85 (0.90 is documented to OOM
on GB10 with this checkpoint), a check that the running container is actually
the GLM image, and a poll of ray status for 2 active nodes instead of the
recipes' fixed ~25s sleep.

enforce-eager defaults ON here: ~12.9 GiB/node of headroom against Nemotron's
roughly double, and graph capture is a spike."
```

---

### Task 5: Make PROFILE mandatory

**Files:**
- Modify: `Makefile`

**Interfaces:**
- Consumes: `glm/cluster-env.sh` (Task 3), `nemotron/cluster-env.sh` (existing).
- Produces: `make head|worker|serve PROFILE=<glm|nemotron>`; bare invocations fail.

- [ ] **Step 1: Write the failing test**

Run: `make head 2>&1 | head -5; echo "exit=${PIPESTATUS[0]}"`
Expected today: it runs the Nemotron bring-up. That is the bug — a bare `make head` must not silently pick a profile.

- [ ] **Step 2: Replace the head/worker/nemotron targets**

In `Makefile`, replace the `head:`, `worker:` and `nemotron:` targets with:

```make
# PROFILE selects the model profile: its cluster-env.sh supplies both the
# VLLM_FORWARD_VARS set and VLLM_IMAGE. There is deliberately NO default.
#
# The two profiles need *different* forwarded vars and different images, so a
# default would let a bare `make head` bring the cluster up on the wrong image
# with the wrong env -- which does not error, it hangs in a collective when
# rank 1 picks a different backend from rank 0. An explicit profile on every
# invocation is cheap; diagnosing a silent wrong-image bring-up is not.
VALID_PROFILES := nemotron glm

require-profile:
	@if [ -z "$(PROFILE)" ]; then \
		echo "PROFILE is required. Valid profiles: $(VALID_PROFILES)" >&2; \
		echo "  e.g.  make $(firstword $(MAKECMDGOALS)) PROFILE=glm" >&2; \
		exit 1; \
	fi
	@if ! echo "$(VALID_PROFILES)" | tr ' ' '\n' | grep -qx "$(PROFILE)"; then \
		echo "Unknown PROFILE '$(PROFILE)'. Valid profiles: $(VALID_PROFILES)" >&2; \
		exit 1; \
	fi
	@if [ ! -r "$(PROFILE)/cluster-env.sh" ]; then \
		echo "PROFILE '$(PROFILE)' has no $(PROFILE)/cluster-env.sh" >&2; \
		exit 1; \
	fi

head: require-profile preflight
	cd cluster/head && . ../../$(PROFILE)/cluster-env.sh && bash run_headnode_2.sh

worker: require-profile preflight
	cd cluster/worker && . ../../$(PROFILE)/cluster-env.sh && bash run_workernode_2.sh

serve: require-profile
	@case "$(PROFILE)" in \
	  nemotron) cd nemotron && bash launch-nemotron-120b.sh ;; \
	  glm)      cd glm && bash launch-glm53-flash.sh ;; \
	esac
```

Update `.PHONY` to include `require-profile` and `serve`, and drop `nemotron`. Update the `help:` target to show `PROFILE=<nemotron|glm>` on the head/worker/serve lines and to state that it is required.

- [ ] **Step 3: Verify a bare invocation now fails**

Run: `make head; echo "exit=$?"`
Expected: `PROFILE is required. Valid profiles: nemotron glm`, `exit=2` (make's exit for a failed recipe; any non-zero is acceptable). It must not reach `check_nvidia.sh` or `run_headnode_2.sh`.

- [ ] **Step 4: Verify a bogus profile fails (Review Focus 1)**

Run: `make head PROFILE=typo; echo "exit=$?"`
Expected: `Unknown PROFILE 'typo'. Valid profiles: nemotron glm`, non-zero exit, and no cluster started.

- [ ] **Step 5: Verify a valid profile still reaches preflight**

Run: `make head PROFILE=nemotron 2>&1 | head -12`
Expected: the NVIDIA driver check output appears (preflight ran), then the Nemotron bring-up begins. Ctrl-C out — this confirms routing, not a full bring-up.

- [ ] **Step 6: Commit**

```bash
git add Makefile
git commit -m "Require PROFILE on every bring-up target

The profiles carry different VLLM_FORWARD_VARS and different images, so a
default would let a bare 'make head' come up on the wrong image with the wrong
env -- and that failure is a rank-1 collective hang, not an error message.

Breaking change: 'make head' and 'make worker' now fail until PROFILE is
given, and 'make nemotron' becomes 'make serve PROFILE=nemotron'. Intended."
```

---

### Task 6: Climb the context ladder (rungs 1–3)

Each rung is a full bring-up, a smoke test, and a teardown. Do not skip to 262K: headroom is ~12.9 GiB/node, and the documented worst case on this hardware needed a power cycle to recover.

**Files:**
- Create: `glm/LADDER.md` (the measurement log)

**Interfaces:**
- Consumes: everything from Tasks 2–5.
- Produces: `glm/LADDER.md` with per-rung `MemAvailable` on both nodes, decode tok/s, and the resolved `REASONING_PARSER` value.

- [ ] **Step 1: Pre-flight both nodes**

Run on each node:
```bash
bash scripts/check_nvidia.sh                    # expect exit 0, same driver version both nodes
docker info 2>/dev/null | grep -i "cgroup driver"   # must say cgroupfs
systemctl is-enabled earlyoom 2>/dev/null || echo "earlyoom NOT enabled -- install it before proceeding"
systemctl show ssh -p OOMScoreAdjust             # expect OOMScoreAdjust=-1000
sudo sysctl -w vm.swappiness=0
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
free -g
```
Expected: driver healthy and identical on both, `cgroupfs`, earlyoom enabled, sshd protected, swappiness 0, ~115 GB free.

Do not proceed if earlyoom or the sshd OOM score are missing. With this headroom they are prerequisites, not hardening.

- [ ] **Step 2: Rung 1 — 32K context, no speculation**

Node 1: `source glm/cluster-env.sh && make head PROFILE=glm`
Node 2: `source glm/cluster-env.sh && make worker PROFILE=glm`
Node 1, new terminal: `MAX_MODEL_LEN=32768 GPU_MEM_UTIL=0.80 bash glm/launch-glm53-flash.sh`

Expected: weights load (first run downloads ~181 GiB — allow time), Ray shows 2 nodes, server binds :8000.

While it loads, watch on both nodes: `watch -n5 'grep MemAvailable /proc/meminfo'`

- [ ] **Step 3: Smoke-test rung 1**

```bash
source glm/.env
curl -s http://localhost:8000/health && echo " health OK"
curl -s http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer ${VLLM_API_KEY}" -H 'Content-Type: application/json' \
  -d '{"model":"zai-org/glm-5.3-flash","messages":[{"role":"user","content":"Reply with exactly: ok"}],"max_tokens":16}' \
  | tee /dev/stderr | grep -q '"content"' && echo "completion OK"
```
Expected: `health OK`, a completion containing `ok`. Record `MemAvailable` on both nodes in `glm/LADDER.md`.

- [ ] **Step 4: Resolve the reasoning-parser conflict**

The sources disagree (`deepseek_r1` vs `glm45`) and a wrong parser does not error — it mis-splits reasoning from content. Probe both against a prompt that forces reasoning:

```bash
source glm/.env
probe() {
  curl -s http://localhost:8000/v1/chat/completions \
    -H "Authorization: Bearer ${VLLM_API_KEY}" -H 'Content-Type: application/json' \
    -d '{"model":"zai-org/glm-5.3-flash","messages":[{"role":"user","content":"What is 17*23? Think step by step."}],"max_tokens":512}' \
    | python3 -c 'import json,sys; d=json.load(sys.stdin)["choices"][0]["message"];
print("reasoning_content:", (d.get("reasoning_content") or "<EMPTY>")[:120]);
print("content:", (d.get("content") or "<EMPTY>")[:120])'
}
probe
```
Run once with the server started under `REASONING_PARSER=deepseek_r1`, then restart it with `REASONING_PARSER=glm45` and run again.

Correct parser: `reasoning_content` holds the step-by-step working and `content` holds just the answer. Wrong parser: `reasoning_content` is empty and the raw thinking leaks into `content`, or vice versa. Record the winner in `glm/LADDER.md` and set it as the default in `glm/launch-glm53-flash.sh`.

- [ ] **Step 5: Rung 2 — 131K context**

Tear down (both nodes: `docker stop node-*`), then bring up and run:
`MAX_MODEL_LEN=131072 GPU_MEM_UTIL=0.85 bash glm/launch-glm53-flash.sh`

Repeat the Step 3 smoke test. Record `MemAvailable` on both nodes.

Abort if `MemAvailable` on either node drops below ~4 GB, sshd latency degrades, or `dmesg -T | tail -30` shows UVM or OOM activity. If it trips, the previous rung is your working configuration — record that and stop.

- [ ] **Step 6: Rung 3 — 262K context, the target**

Tear down, bring up, then:
`MAX_MODEL_LEN=262144 GPU_MEM_UTIL=0.85 bash glm/launch-glm53-flash.sh`

Smoke test, then a long-context check that actually exercises the KV budget:
```bash
source glm/.env
python3 - <<'PY'
import json, subprocess, os
prompt = "The magic word is 'zarquon'. " + ("filler text. " * 20000) + " What is the magic word?"
body = json.dumps({"model":"zai-org/glm-5.3-flash",
                   "messages":[{"role":"user","content":prompt}],"max_tokens":32})
out = subprocess.run(["curl","-s","http://localhost:8000/v1/chat/completions",
  "-H",f"Authorization: Bearer {os.environ['VLLM_API_KEY']}",
  "-H","Content-Type: application/json","-d",body],capture_output=True,text=True).stdout
print(out[:600])
PY
```
Expected: the response recovers `zarquon`, demonstrating a long prompt survives the fp8 KV budget rather than silently truncating. Record `MemAvailable` at peak.

- [ ] **Step 7: Measure baseline decode speed**

```bash
source glm/.env
time curl -s http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer ${VLLM_API_KEY}" -H 'Content-Type: application/json' \
  -d '{"model":"zai-org/glm-5.3-flash","messages":[{"role":"user","content":"Write a Python function that merges two sorted lists. Explain it."}],"max_tokens":400}' \
  | python3 -c 'import json,sys; u=json.load(sys.stdin)["usage"]; print(u)'
```
Record completion tokens and wall time in `glm/LADDER.md` — this is the no-speculation baseline that Task 7's DFlash2 number is measured against.

- [ ] **Step 8: Commit the ladder log**

```bash
git add glm/LADDER.md glm/launch-glm53-flash.sh
git commit -m "Record context ladder rungs 1-3 and resolve the reasoning parser

MemAvailable and decode baselines per rung on both nodes, and the probed
answer to the deepseek_r1-vs-glm45 disagreement -- a wrong reasoning parser
does not error, it mis-splits reasoning_content from content, so it had to be
observed rather than chosen."
```

---

### Task 7: Enable DFlash2 and measure

**Files:**
- Modify: `glm/launch-glm53-flash.sh` (fill in `GLM_SPEC_CONFIG` from `glm/DISCOVERY.md`)
- Modify: `glm/LADDER.md` (rung 4)

**Interfaces:**
- Consumes: `GLM_SPEC_CONFIG` recorded in `glm/DISCOVERY.md` (Task 3), the rung-3 baseline (Task 6).
- Produces: a measured tok/s and acceptance rate against the published 46.9 tok/s / 74.1%.

- [ ] **Step 1: Wire the discovered spec config into the launcher**

Replace the `GLM_SPEC_CONFIG` reference in `glm/launch-glm53-flash.sh` with the exact value from `glm/DISCOVERY.md`, as a default that stays overridable:

```bash
SPEC_FLAG=""
if [[ "${ENABLE_DFLASH2}" == "1" ]]; then
  # Exact spelling and JSON from glm/DISCOVERY.md, read off the image's own
  # launcher. LICENCE: the drafter is CC-BY-NC-ND-4.0 -- research use only.
  SPEC_FLAG="${GLM_SPEC_CONFIG:-<exact value from DISCOVERY.md>}"
fi
```

- [ ] **Step 2: Rung 4 — 262K with DFlash2**

Tear down, bring up, then:
`MAX_MODEL_LEN=262144 GPU_MEM_UTIL=0.85 ENABLE_DFLASH2=1 bash glm/launch-glm53-flash.sh`

Expected: the drafter downloads and loads; the server binds. Watch `MemAvailable` — the drafter slot-shares MLA tensors and should add no KV cost, so a large drop here is a signal something is wrong.

- [ ] **Step 3: Measure decode speed and acceptance**

```bash
source glm/.env
time curl -s http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer ${VLLM_API_KEY}" -H 'Content-Type: application/json' \
  -d '{"model":"zai-org/glm-5.3-flash","messages":[{"role":"user","content":"Write a Python function that merges two sorted lists. Explain it."}],"max_tokens":400}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["usage"])'

curl -s http://localhost:8000/metrics | grep -Ei 'spec_decode|draft|accept' || \
  echo "(no spec-decode metrics exposed; use the server log's acceptance line)"
```
Expected: decode tok/s materially above the Task 6 Step 7 baseline. Published reference is 46.9 tok/s at 74.1% acceptance; treat a large shortfall as a finding to record, not a failure to hide.

- [ ] **Step 4: Record the comparison**

Append to `glm/LADDER.md`: baseline vs DFlash2 tok/s, the measured acceptance rate, peak `MemAvailable` on both nodes, and how the numbers compare to the published figures. If they diverge substantially, note it plainly — the published figures came from a different bring-up path (Ray-less rank launch) and may not transfer exactly.

- [ ] **Step 5: Decide and record the default**

If DFlash2 is stable and faster, flip `ENABLE_DFLASH2` to default `1` in the launcher. If it is unstable or the gain is small, leave the default `0` and record why. Either way the reasoning goes in `glm/LADDER.md`.

- [ ] **Step 6: Commit**

```bash
git add glm/launch-glm53-flash.sh glm/LADDER.md
git commit -m "Enable and measure DFlash2 speculative decoding

Measured against the rung-3 no-speculation baseline on this cluster rather
than assuming the published 46.9 tok/s / 74.1%, which came from a Ray-less
rank-launch path. Drafter stays CC-BY-NC-ND-4.0: research use only."
```

---

### Task 8: Document the path in CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: everything above.
- Produces: no code.

- [ ] **Step 1: Add the GLM section to CLAUDE.md**

After the Nemotron section, add a `## GLM-5.3-Flash-NVFP4 (Ray TP=2 across both Sparks)` section covering: the checkpoint and what weight-only NVFP4-A16 means; the memory budget arithmetic and why 0.85 is a ceiling rather than a starting point; why the image is `local/vllm-ray-glm53:sm121-v11-dflash2` layered on a digest-pinned GHCR base, and that the base ships no Ray; that stock vLLM cannot run this model on GB10 because of NoPE MLA, so the patched image is not optional; `--tool-call-parser glm47` and the probed reasoning-parser answer; the one-model-at-a-time UMA invariant; and the DFlash2 licence constraint.

- [ ] **Step 2: Update the bring-up sequence section**

The existing "Bring-up sequence (must run in order)" section shows raw `run_*node_2.sh` invocations and no profile. Update it to the `PROFILE`-based flow and state that `PROFILE` is mandatory with no default, including the reason: the profiles carry different forwarded vars and different images, and a wrong-image bring-up hangs in a collective rather than erroring.

- [ ] **Step 3: Update the driver-lockstep section**

Add the 2026-09-27 `pulsar` instance beside the 2026-08-20 one, and note how it differed: not metapackage-vs-metapackage drift but a pinpoint `linux-image-7.0.0-1019-nvidia` installed ahead of the metapackage pair, which is why `check_nvidia.sh` read `metapkg lockstep OK` throughout the outage. Note the new `metapkg covers kernel` row added in Task 1.

- [ ] **Step 4: Update the Health / monitoring section**

Add `scripts/test_nvidia_lib.sh` (unit tests for the pure ABI helpers) and `glm/verify-image.sh` (the Ray-layer pin gate) to the list.

- [ ] **Step 5: Update README.md**

Add GLM to the models list and the `PROFILE`-based commands.

- [ ] **Step 6: Verify the docs match reality**

Run: `make help`
Expected: every command shown in CLAUDE.md's bring-up section exists in the help output with the same spelling, and `PROFILE` is shown as required.

- [ ] **Step 7: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "Document the GLM-5.3-Flash path and the PROFILE requirement

Also records the 2026-09-27 driver outage beside the 2026-08-20 one: a
pinpoint kernel installed ahead of the metapackage pair, which the old
lockstep check could not see because it compared the metapackages only to
each other."
```

---

## Notes for the executor

- **Never bring Ray up with the two nodes on different driver versions.** Re-run `scripts/check_nvidia.sh` on both after any `apt upgrade`.
- **Tear down both ranks between rungs.** Stale Ray/NCCL processes on either node fight the next start. `docker stop node-*` on both; Ctrl-C on the head stops only the head container.
- **A warm reboot sheds the f1 half of each CX7 port.** Bring-up still works on 2 links; only a cold boot (AC removed) restores all 4. Do not "fix" this by pinning HCAs in a profile.
- **If `docker info` stops saying `cgroupfs`**, stop and fix it before anything else — a `daemon-reload` under the systemd driver silently revokes the container's GPU device-cgroup permission, and the symptom (`CUDA error: operation not permitted`, often at `capture_end`) looks like a model bug.
- **Report measured numbers as measured**, including shortfalls against the published figures. The published numbers came from a different bring-up path.
