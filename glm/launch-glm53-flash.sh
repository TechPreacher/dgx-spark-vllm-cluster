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
# head stay BF16. The checkpoint declares the MULTIMODAL architecture
# (Glm5NextForConditionalGeneration), which is why --skip-mm-profiling matters
# even for a text-only run: the vision tower is in the graph regardless.
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
# LICENCE: the DFlash2 drafter (incoai/GLM-5.3-Flash-DFlash2) is
# CC-BY-NC-ND-4.0 -- research / personal use only. Do not redistribute it and do
# not bake it into a shared image. The target model itself is MIT. Leave
# ENABLE_DFLASH2=0 for a licence-clean run.
#
# See glm/DISCOVERY.md for how every flag and env var below was established.

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
DRAFT_CKPT="${DRAFT_CKPT:-incoai/GLM-5.3-Flash-DFlash2}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-7}"
# Recipes disagree: the checkpoint card says deepseek_r1, one 2-Spark recipe
# says glm45. A wrong parser does not error -- it silently mis-splits
# reasoning_content from content -- so this is probed at ladder rung 1.
REASONING_PARSER="${REASONING_PARSER:-deepseek_r1}"
# This vLLM build defaults distributed_executor_backend to "mp" (config/parallel.py
# :917) and does NOT infer "ray" from a live Ray cluster the way the NGC 26.05
# build behind the Nemotron path does. Without this flag, multiprocessing sees one
# local GPU and refuses world size 2 outright:
#   "World size (2) is larger than the number of available GPUs (1) in this node."
# Accepted values: ray | mp | uni | external_launcher.
DIST_BACKEND="${DIST_BACKEND:-ray}"
# The patched image's Glm5NextProcessor.from_pretrained does a raw
#   open(os.path.join(model_path, "processor_config.json"))
# (transformers_utils/processors/glm5next.py:853) instead of resolving through
# the Hub, so it ONLY works when --model is a local directory. Passed a repo id
# it dies with FileNotFoundError on a file it has already downloaded into the
# cache. So resolve the repo id to its snapshot directory before serving.
# Set MODEL_PATH to skip resolution and use a directory directly.
MODEL_PATH="${MODEL_PATH:-}"
EXPECTED_IMAGE="${EXPECTED_IMAGE:-local/vllm-ray-glm53:sm121-v11-dflash2}"
EXPECTED_BASE_DIGEST="${EXPECTED_BASE_DIGEST:-$(cat "${SCRIPT_DIR}/BASE_DIGEST")}"

# Refuse to exceed the documented OOM ceiling, however the caller was invoked.
# mem_util_within_ceiling (cluster/lib.sh) validates the format FIRST and fails
# closed: the previous inline awk interpolated the value into program text, so
# "inf", "0,85" and "abc" all passed the guard.
if ! mem_util_within_ceiling "${GPU_MEM_UTIL}" 0.85; then
  echo "ERROR: GPU_MEM_UTIL=${GPU_MEM_UTIL} is not an accepted value (must be a" >&2
  echo "decimal <= 0.85). 0.90 is documented to OOM on GB10 with this checkpoint." >&2
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

# Beyond the image NAME, assert the base digest recorded at build time. A tag can
# be rebuilt from a different base; glm/verify-image.sh checks this at build time
# but nothing stopped a later rebuild, and the label costs one inspect here.
RUNNING_BASE=$(docker image inspect "${RUNNING_IMAGE}" \
  --format '{{index .Config.Labels "glm.base.digest"}}' 2>/dev/null || true)
if [[ "${RUNNING_BASE}" != "${EXPECTED_BASE_DIGEST}" ]]; then
  cat >&2 <<EOF
ERROR: ${RUNNING_IMAGE} was not built from the pinned base digest.
  recorded: ${RUNNING_BASE:-<none>}
  expected: ${EXPECTED_BASE_DIGEST}

Rebuild it and re-run the gate on BOTH nodes:
  BASE_IMAGE=${EXPECTED_BASE_DIGEST} TAG=${EXPECTED_IMAGE} bash cluster/build-image.sh
  make verify-glm-image
EOF
  exit 1
fi

echo "Using container: ${VLLM_CONTAINER}  (${RUNNING_IMAGE})"
echo "  model:             ${MODEL_CKPT}"
echo "  TP:                ${TP_SIZE}"
echo "  executor backend:  ${DIST_BACKEND}"
echo "  max-model-len:     ${MAX_MODEL_LEN}"
echo "  gpu-mem-util:      ${GPU_MEM_UTIL}"
echo "  kv-cache-memory:   ${KV_CACHE_MEMORY}"
echo "  block-size:        ${BLOCK_SIZE}"
echo "  max-num-seqs:      ${MAX_NUM_SEQS}"
echo "  enforce-eager:     ${ENABLE_EAGER}"
echo "  DFlash2 spec:      ${ENABLE_DFLASH2}"
echo "  reasoning parser:  ${REASONING_PARSER}"
echo "  port:              ${PORT}"

# Same guard as the Nemotron launcher: if the forwarded vars are absent from the
# head container's env, the user did not source glm/cluster-env.sh before
# bring-up. Rank 1 on the worker will not have them either, and the run hangs in
# a collective rather than erroring. Fail here with the fix instead.
FORWARD_VARS=$(bash -c 'source '"${SCRIPT_DIR}"'/cluster-env.sh >/dev/null 2>&1; echo "${VLLM_FORWARD_VARS}"')
# An empty list makes the loop below iterate zero times, so the guard would pass
# silently -- which is what happened if cluster-env.sh was missing or errored.
# The glm profile always has at least one forwarded var, so empty means broken.
if [[ -z "${FORWARD_VARS// /}" ]]; then
  echo "ERROR: could not read VLLM_FORWARD_VARS from ${SCRIPT_DIR}/cluster-env.sh." >&2
  echo "Refusing to serve: without it the env guard below cannot check anything." >&2
  exit 1
fi
# || true so a docker failure surfaces as a named cause rather than an
# empty-handed `set -e` exit with no output at all.
MISSING_VARS=$(docker exec "${VLLM_CONTAINER}" /bin/bash -c '
  set -u
  missing=""
  for V in '"${FORWARD_VARS}"'; do
    [[ -z "${!V:-}" ]] && missing="${missing} $V"
  done
  echo "${missing}"
' 2>/dev/null | xargs) || true
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

# Readiness: BOTH two active nodes and two GPUs. Parsing is delegated to the
# pure helpers in cluster/lib.sh (unit-tested by scripts/test_cluster_lib.sh).
#
# Two nodes AND two GPUs, not just two GPUs: if `make worker` is run on Node 1 by
# mistake it joins itself, Ray reports 2 GPUs, and vLLM places both TP shards on
# one Spark -- 181 GiB onto a single 128 GB host with no cgroup cap.
echo -n "Waiting for Ray to report 2 nodes and 2 GPUs"
NODES=0
GPUS=0
for _ in $(seq 1 60); do
  # || true: a transient docker exec failure must not kill the poll silently.
  STATUS=$(docker exec "${VLLM_CONTAINER}" ray status 2>/dev/null || true)
  NODES=$(ray_node_count_from_status "${STATUS}")
  GPUS=$(ray_gpu_total_from_status "${STATUS}")
  if [[ "${NODES}" -ge 2 && "${GPUS}" -ge 2 ]]; then
    echo " -- ${NODES} nodes, ${GPUS} GPUs"
    break
  fi
  echo -n "."
  sleep 2
done
if [[ "${NODES}" -lt 2 || "${GPUS}" -lt 2 ]]; then
  echo
  echo "ERROR: Ray reports ${NODES} node(s) and ${GPUS} GPU(s); need 2 and 2." >&2
  if [[ "${NODES}" -ge 2 && "${GPUS}" -lt 2 ]]; then
    echo "Two nodes but too few GPUs -- check the driver on the worker:" >&2
    echo "  bash scripts/check_nvidia.sh      # on Node 2" >&2
  elif [[ "${NODES}" -lt 2 && "${GPUS}" -ge 2 ]]; then
    echo "Only one node is providing GPUs. Is the worker running on Node 1 by" >&2
    echo "mistake? Both TP shards would land on one Spark. Tear down and restart" >&2
    echo "the worker on Node 2." >&2
  else
    echo "The worker has not joined. On Node 2:" >&2
    echo "  source glm/cluster-env.sh && make worker PROFILE=glm" >&2
  fi
  exit 1
fi

EAGER_FLAG=""
[[ "${ENABLE_EAGER}" == "1" ]] && EAGER_FLAG="--enforce-eager"

# Method is "dflash" -- NOT "dflash2". The 2 lives in the drafter's architecture
# (DFlash2DraftModel), not in vLLM's method string: config/speculative.py has
# DFlashModelTypes = Literal["dflash"]. vLLM derives n_predict from the drafter's
# block_size (8) when unset, and sets parallel_drafting=True for dflash.
# See glm/DISCOVERY.md.
SPEC_FLAG=""
if [[ "${ENABLE_DFLASH2}" == "1" ]]; then
  if [[ -n "${GLM_SPEC_CONFIG:-}" ]]; then
    SPEC_FLAG="${GLM_SPEC_CONFIG}"
  else
    SPEC_FLAG=$(printf '{"method":"dflash","model":"%s","num_speculative_tokens":%s}' \
                  "${DRAFT_CKPT}" "${NUM_SPEC_TOKENS}")
  fi
  echo "  speculative:       ${SPEC_FLAG}"
  echo "  NOTE: ${DRAFT_CKPT} is CC-BY-NC-ND-4.0 -- research use only."
fi

docker exec -it \
  -e VLLM_API_KEY="${VLLM_API_KEY}" \
  -e HF_TOKEN="${HF_TOKEN}" \
  -e MODEL_CKPT="${MODEL_CKPT}" \
  -e MODEL_PATH="${MODEL_PATH}" \
  -e SKIP_WEIGHT_CHECK="${SKIP_WEIGHT_CHECK:-0}" \
  -e SERVED_NAME="${SERVED_NAME}" \
  -e MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
  -e GPU_MEM_UTIL="${GPU_MEM_UTIL}" \
  -e KV_CACHE_MEMORY="${KV_CACHE_MEMORY}" \
  -e BLOCK_SIZE="${BLOCK_SIZE}" \
  -e MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
  -e TP_SIZE="${TP_SIZE}" \
  -e DIST_BACKEND="${DIST_BACKEND}" \
  -e PORT="${PORT}" \
  -e EAGER_FLAG="${EAGER_FLAG}" \
  -e SPEC_FLAG="${SPEC_FLAG}" \
  -e REASONING_PARSER="${REASONING_PARSER}" \
  "${VLLM_CONTAINER}" /bin/bash -c '
    set -euo pipefail

    # Resolve the checkpoint to a LOCAL directory (see MODEL_PATH note above).
    # snapshot_download is resumable and writes into the bind-mounted HF cache,
    # so an interrupted pull continues rather than restarting.
    if [[ -n "${MODEL_PATH}" ]]; then
      MODEL_DIR="${MODEL_PATH}"
    elif [[ "${MODEL_CKPT}" == /* || "${MODEL_CKPT}" == .* ]]; then
      MODEL_DIR="${MODEL_CKPT}"
    else
      echo "Resolving ${MODEL_CKPT} to a local snapshot (~181 GiB on first run)..."
      MODEL_DIR=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('"'"'${MODEL_CKPT}'"'"'))")
      echo "Model directory: ${MODEL_DIR}"
    fi
    if [[ ! -f "${MODEL_DIR}/processor_config.json" ]]; then
      echo "ERROR: ${MODEL_DIR}/processor_config.json missing -- the glm5next" >&2
      echo "processor reads it by path and will fail without it." >&2
      exit 1
    fi

    # EVERY node must have the checkpoint, not just this one. The HF cache is
    # bind-mounted per node, and each rank loads its shard from its own
    # filesystem. Without this check, a worker missing the weights surfaces ~30s
    # into engine init as a Ray traceback that names the path but not the cause.
    # Probe via Ray so we see exactly the nodes Ray will schedule on, through the
    # same mount the workers use -- no ssh, no assumptions about host names.
    if [[ "${SKIP_WEIGHT_CHECK}" == "1" ]]; then
      echo "Skipping the per-node checkpoint check (SKIP_WEIGHT_CHECK=1)."
    else
    echo "Checking every Ray node has the checkpoint..."
    python3 - <<"PY"
import os, sys, glob, socket
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

path = os.environ["MODEL_DIR"]
ray.init(address="auto", logging_level="ERROR")

@ray.remote(num_cpus=0)
def probe(p):
    return (socket.gethostname(),
            os.path.isdir(p),
            len(glob.glob(os.path.join(p, "*.safetensors"))))

nodes = [n for n in ray.nodes() if n.get("Alive")]
refs = [probe.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=n["NodeID"], soft=False)).remote(path) for n in nodes]

bad = []
for host, is_dir, n_files in ray.get(refs):
    status = "ok" if (is_dir and n_files) else "MISSING"
    print(f"  {host:<12} dir={is_dir} safetensors={n_files}  {status}")
    if not (is_dir and n_files):
        bad.append(host)

if bad:
    joined = ", ".join(bad)
    print("", file=sys.stderr)
    print("ERROR: checkpoint missing on: " + joined, file=sys.stderr)
    print("Ray TP loads each shard from its own node filesystem, and the HF", file=sys.stderr)
    print("cache is per node. Run this on the affected node(s):", file=sys.stderr)
    print("  bash glm/fetch-weights.sh", file=sys.stderr)
    raise SystemExit(1)
PY
    fi

    SPEC_ARGS=()
    [[ -n "${SPEC_FLAG}" ]] && SPEC_ARGS+=(--speculative-config "${SPEC_FLAG}")
    # shellcheck disable=SC2086
    exec vllm serve "${MODEL_DIR}" \
      --served-model-name "${SERVED_NAME}" \
      --host 0.0.0.0 \
      --port "${PORT}" \
      --tensor-parallel-size "${TP_SIZE}" \
      --distributed-executor-backend "${DIST_BACKEND}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --gpu-memory-utilization "${GPU_MEM_UTIL}" \
      --kv-cache-dtype fp8 \
      --kv-cache-memory-bytes "${KV_CACHE_MEMORY}" \
      --block-size "${BLOCK_SIZE}" \
      --max-num-seqs "${MAX_NUM_SEQS}" \
      --enable-auto-tool-choice \
      --tool-call-parser glm47 \
      --reasoning-parser "${REASONING_PARSER}" \
      --skip-mm-profiling \
      ${EAGER_FLAG} \
      ${SPEC_ARGS[@]+"${SPEC_ARGS[@]}"}
  '
