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

# LC_ALL=C on BOTH sides is load-bearing, not cosmetic. `sort` runs inside the
# container and `comm` runs on the host, and the two disagree on collation:
# en_US.utf8 orders "aiohttp Beta Zlib", C orders "Beta Zlib aiohttp". Without a
# pinned collation, comm rejects its own correctly-sorted input with
# "file N is not in sorted order" -- which it did on the first run of this script.
freeze() {
  docker run --rm --entrypoint /bin/bash "$1" -c \
    'pip freeze --disable-pip-version-check 2>/dev/null | sed "/^-e /d" | LC_ALL=C sort'
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
CHANGED=$(LC_ALL=C comm -23 "${TMP}/base.txt" "${TMP}/layered.txt")
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

ADDED=$(LC_ALL=C comm -13 "${TMP}/base.txt" "${TMP}/layered.txt" | wc -l)
echo "additions   ${ADDED} package(s) added by the ray layer"

if ! docker run --rm --entrypoint /bin/bash "${LAYERED_TAG}" -c 'ray --version' >/dev/null 2>&1; then
  echo "FAIL: ray CLI is not callable in ${LAYERED_TAG}." >&2
  exit 1
fi
echo "ray         OK ($(docker run --rm --entrypoint /bin/bash "${LAYERED_TAG}" -c 'ray --version' 2>&1 | head -n1))"

echo
echo "OK  ${LAYERED_TAG} is safe to use on $(hostname)."
