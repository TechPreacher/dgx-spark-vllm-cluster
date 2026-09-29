#!/usr/bin/env bash
set -uo pipefail
# Static assertions over the flags glm/launch-glm53-flash.sh passes to vllm serve.
#
# These are cheap guards against a class of bug that has already bitten twice on
# this launcher, both times only discoverable at bring-up time after the cluster
# was already running:
#   * --kv-cache-memory was not a registered flag (real name:
#     --kv-cache-memory-bytes); it resolved only via argparse abbreviation.
#   * --distributed-executor-backend was missing entirely, so vLLM defaulted to
#     "mp", saw one local GPU and refused world size 2.
#
# Run: bash scripts/test_glm_launcher.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${SCRIPT_DIR}/../glm/launch-glm53-flash.sh"

PASS=0
FAIL=0
present() {
  if grep -qF -- "$2" "${LAUNCHER}"; then
    PASS=$((PASS + 1)); printf '  ok    %s\n' "$1"
  else
    FAIL=$((FAIL + 1)); printf '  FAIL  %s (missing: %s)\n' "$1" "$2"
  fi
}
absent() {
  if grep -qE -- "$2" "${LAUNCHER}"; then
    FAIL=$((FAIL + 1)); printf '  FAIL  %s (found: %s)\n' "$1" "$2"
  else
    PASS=$((PASS + 1)); printf '  ok    %s\n' "$1"
  fi
}

echo
echo "glm launcher flag assertions"
echo

present "ray executor backend is passed explicitly" "--distributed-executor-backend"
# The image's auto choice (flashinfer_cutlass) JIT-builds a module that cannot
# compile here; marlin is prebuilt. Must be passed explicitly, not left to auto.
present "moe backend is passed explicitly"             "--moe-backend"
present "kv cache memory uses the registered flag name" "--kv-cache-memory-bytes"
present "tool parser pinned to glm47"                  "--tool-call-parser glm47"
present "multimodal profiling skipped"                 "--skip-mm-profiling"
present "fp8 kv cache"                                 "--kv-cache-dtype fp8"
present "tensor parallel size is passed"               "--tensor-parallel-size"
# The glm5next processor opens processor_config.json by path, so a repo id must
# be resolved to a local snapshot dir before it reaches vllm serve.
present "checkpoint resolved to a local dir"           "snapshot_download"
present "serve is given the resolved dir, not the id"  'vllm serve "${MODEL_DIR}"'
absent  "serve is not given the raw repo id"           'vllm serve "\$\{MODEL_CKPT\}"' 

# The abbreviated form must not come back: it resolves today but breaks the moment
# another --kv-cache-memory* option is registered, and it breaks at model launch.
# The probe reads MODEL_DIR via os.environ, so a bare shell assignment is not
# enough -- this exact omission cost a run.
present "MODEL_DIR is exported for the probe"           "export MODEL_DIR"
absent  "no abbreviated --kv-cache-memory"             '\-\-kv-cache-memory "'
# Both sources warn against these two for this model generation.
absent  "tool parser is not glm or glm45"              '\-\-tool-call-parser (glm|glm45) '

echo
echo "  ${PASS} passed, ${FAIL} failed"
echo
[[ "${FAIL}" -eq 0 ]]
