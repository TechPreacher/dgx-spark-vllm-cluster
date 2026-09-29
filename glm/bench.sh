#!/usr/bin/env bash
set -euo pipefail

# Measure decode throughput against a running GLM server, the same way every time.
#
# Exists because this deployment has several configurations to compare and the
# comparison is only meaningful if the prompt, sampling and token budget are
# identical: marlin vs flashinfer_cutlass vs DFlash2 speculation. Eyeballing one
# curl against another is how you end up believing a difference that was really
# prefix caching or a different max_tokens.
#
# Usage:
#   bash glm/bench.sh                 # 3 runs, label taken from the server
#   RUNS=5 LABEL=cutlass bash glm/bench.sh
#
# Reports per-run decode tok/s and the median. Writes nothing; paste the result
# into glm/LADDER.md yourself so the log stays a deliberate record.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../cluster/lib.sh
source "${SCRIPT_DIR}/../cluster/lib.sh"
load_env "${SCRIPT_DIR}"
: "${VLLM_API_KEY:?VLLM_API_KEY not set (expected in glm/.env)}"

PORT="${PORT:-8000}"
RUNS="${RUNS:-3}"
MAX_TOKENS="${MAX_TOKENS:-400}"
LABEL="${LABEL:-unlabelled}"
SERVED_NAME="${SERVED_NAME:-zai-org/glm-5.3-flash}"

# A prompt long enough to produce a few hundred tokens of real work, and varied
# per run so prefix caching cannot make run 2 look faster than run 1.
if ! curl -fsS -m 10 "http://localhost:${PORT}/health" >/dev/null 2>&1; then
  echo "ERROR: no healthy server on :${PORT}" >&2
  exit 1
fi

echo "benchmark: ${LABEL}   runs=${RUNS}  max_tokens=${MAX_TOKENS}"
echo

declare -a RATES=()
for i in $(seq 1 "${RUNS}"); do
  # Vary the prompt so each run does real prefill+decode, not a cache hit.
  PROMPT="Run ${i}: write a Python function that merges two sorted lists, then explain how it works step by step."
  BODY=$(python3 -c '
import json, sys
print(json.dumps({"model": sys.argv[1], "messages": [{"role": "user", "content": sys.argv[2]}],
                  "max_tokens": int(sys.argv[3]), "temperature": 0}))' \
    "${SERVED_NAME}" "${PROMPT}" "${MAX_TOKENS}")

  START=$(date +%s.%N)
  RESP=$(curl -s -m 900 "http://localhost:${PORT}/v1/chat/completions" \
    -H "Authorization: Bearer ${VLLM_API_KEY}" \
    -H 'Content-Type: application/json' -d "${BODY}")
  END=$(date +%s.%N)

  READING=$(python3 -c '
import json, sys
resp, start, end = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
try:
    d = json.loads(resp)
except Exception:
    print("ERR non-JSON"); raise SystemExit(0)
if "error" in d:
    print("ERR " + json.dumps(d["error"])[:120]); raise SystemExit(0)
u = d["usage"]; el = end - start
ct = u["completion_tokens"]
print("%d %.2f %.2f" % (ct, el, ct/el if el > 0 else 0))' "${RESP}" "${START}" "${END}")

  case "${READING}" in
    ERR*) echo "  run ${i}: ${READING}"; continue ;;
  esac
  read -r CT EL RATE <<<"${READING}"
  printf '  run %d: %4s tokens in %6ss  -> %6s tok/s\n' "${i}" "${CT}" "${EL}" "${RATE}"
  RATES+=("${RATE}")
done

echo
if [[ "${#RATES[@]}" -eq 0 ]]; then
  echo "no successful runs" >&2
  exit 1
fi
printf '%s\n' "${RATES[@]}" | sort -n | awk -v l="${LABEL}" '
  {a[NR]=$1}
  END {
    m = (NR%2) ? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2
    printf "MEDIAN (%s): %.1f tok/s   over %d run(s)\n", l, m, NR
  }'
echo
echo "host MemAvailable: $(awk '/MemAvailable/{printf "%.1f GB", $2/1048576}' /proc/meminfo) (this node)"
