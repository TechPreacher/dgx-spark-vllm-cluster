# Nemotron-3-Super-120B-A12B-NVFP4 on the 2-Spark Ray cluster

| Script | Model | TP | ctx | Quant |
|---|---|---|---|---|
| `launch-nemotron-120b.sh` | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | 2 | 1048576 | NVFP4 (native SM121 FP4 tensor cores) |

LatentMoE hybrid (Mamba-2 + MoE + Attention). 120B total / 12B active. Reasoning parser plugin (`super_v3_reasoning_parser.py`) is fetched into the container on first launch.

## One-time setup

```bash
cp .env.example .env
$EDITOR .env                    # set HF_TOKEN and VLLM_API_KEY
# Optional: pre-stage weights (~60-80 GB) before the first run.
hf download nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
```

## Launch (order matters)

**Bring the cluster up with the NVFP4 env profile sourced on each node.** Ray does not propagate `VLLM_*` runtime flags from the head driver to worker ranks across nodes; if the worker container's env lacks these four vars, rank 1 picks a different FP4 GEMM kernel / allreduce backend than rank 0 and the run crashes at the first matmul or collective.

```bash
# Node 1 (head)
source cluster-env.sh
cd ../cluster/head && bash run_headnode_2.sh

# Node 2 (worker)
source cluster-env.sh
cd ../cluster/worker && bash run_workernode_2.sh

# Node 1, new terminal
cd ../nemotron
./launch-nemotron-120b.sh
```

The launcher refuses to start if the four NVFP4 vars are missing inside the head container; it prints these same instructions and exits 1.

## Overrides (env)

```bash
MAX_MODEL_LEN=524288  ./launch-nemotron-120b.sh   # halfway fallback (was the known-stable step)
MAX_MODEL_LEN=262144  ./launch-nemotron-120b.sh   # conservative floor
GPU_MEM_UTIL=0.85     ./launch-nemotron-120b.sh   # raise from the default 0.75
ENABLE_MTP=1          ./launch-nemotron-120b.sh   # MTP speculative decoding
ENABLE_EAGER=1        ./launch-nemotron-120b.sh   # disable CUDA graphs (stability over speed)
```

Other knobs: `MODEL_CKPT`, `SERVED_NAME`, `TP_SIZE`, `PP_SIZE`, `PORT`, `MAMBA_SSM_DTYPE`.

### vLLM-version-dependent flags (defaults match `local/vllm-ray:26.05.post1`)

The launcher turns the following flags ON by default, matching NVIDIA's HF model card for the `local/vllm-ray:26.05.post1` cluster image (built from `nvcr.io/nvidia/vllm:26.05.post1-py3` via `cluster/Dockerfile`). If you fall back to an older image (e.g. `25.11-py3`) and one of them errors, opt out:

```bash
ENABLE_REASONING_PARSER=0  ./launch-nemotron-120b.sh   # drops --reasoning-parser-plugin + --reasoning-parser super_v3
ENABLE_ASYNC_SCHEDULING=0  ./launch-nemotron-120b.sh   # drops --async-scheduling
MOE_BACKEND=               ./launch-nemotron-120b.sh   # drops --moe-backend (vLLM picks default)
CUDAGRAPH_CAPTURE_SIZE=    ./launch-nemotron-120b.sh   # drops --max-cudagraph-capture-size
```

With reasoning parser off, thinking tokens come back in `message.content` rather than being split into a `reasoning_content` field.

## Smoke test

```bash
curl -fsS http://<head-ip>:8000/health && echo
curl http://<head-ip>:8000/v1/chat/completions \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model":"nvidia/nemotron-3-super",
    "messages":[{"role":"user","content":"Explain Lagrange multipliers in 2 sentences."}],
    "max_tokens":500,
    "extra_body":{"chat_template_kwargs":{"enable_thinking":true}}
  }'
```

## Host-stability (mandatory)

The single-Spark `gpt-oss-120b` precedent on this hardware starved the host so badly that sshd became unreachable while ICMP still replied; recovery needed a power cycle. Conservative defaults in the launcher reduce the risk, but the Ray container has no `--memory` cgroup cap, so host-side hardening is **required** on both nodes:

```bash
sudo systemctl edit ssh         # add: [Service]\nOOMScoreAdjust=-1000
sudo apt install earlyoom && sudo systemctl enable --now earlyoom
# External watchdog: curl :8000/health every 30s from a laptop; IPMI/PDU-cycle on N failures.
```

## Shutdown

`Ctrl-C` the launcher (stops vLLM only). To drop the Ray cluster: `Ctrl-C` the head terminal, then on the worker `docker stop $(docker ps --format '{{.Names}}' | grep '^node-')`.
