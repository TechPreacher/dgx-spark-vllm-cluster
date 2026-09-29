# GLM-5.3-Flash (NVFP4) — Ray TP=2 across both Sparks

`zai-org/GLM-5.3-Flash` served as weight-only NVFP4 at tensor-parallel 2, via
`LibertAIDAI/GLM-5.3-Flash-NVFP4`. 320B total / 18B active MoE, natively
multimodal, hybrid sparse + linear attention with Manifold-Constrained
Hyper-Connections. Target model is MIT.

Every flag and env var here was derived by inspecting the image and the
checkpoints — see [DISCOVERY.md](DISCOVERY.md) for the commands and what they
returned, including three places the published recipes were wrong.

## Licence — read before enabling speculation

The target model is MIT. The **DFlash2 drafter
(`incoai/GLM-5.3-Flash-DFlash2`) is CC-BY-NC-ND-4.0**: research and personal
use only, no redistribution, and never baked into a shared image.

As of 2026-09-29 it is **ON by default** (`ENABLE_DFLASH2=1`), because rung 4
measured a **2.8x decode speedup** (40.6 tok/s warm vs 14.4). **This means a
default run is NOT licence-clean.** For a licence-clean run set
`ENABLE_DFLASH2=0`. If this deployment ever needs to serve commercial traffic
the drafter must go — the licence-clean alternative is MTP, which requires the
`RedHatAI/GLM-5.3-Flash-NVFP4` checkpoint instead of this one.

## Why this profile needs its own image

Stock vLLM **cannot run this model on GB10 at all**. The model uses NoPE MLA
(`qk_rope_head_dim=0`) and the stock sparse-attention kernel assumes DeepSeek's
`pe_dim=64`. The patched image extends vLLM's SM90 NoPE sparse-MLA backend to
SM121 via the FA2 path, and carries several other day-0 SM121 fixes — FlashInfer
pinned to 0.6.18 (0.6.17 produced NaN at batch 64–256 rows), NCCL 2.30.7,
CUTLASS 4.6.2, Programmatic Dependent Launch gated off, and an FA2 fp8-KV tile
cap that is what makes fp8 KV cache usable here at all.

That image ships no Ray, so the serving image is `cluster/Dockerfile`'s existing
`ray[default]` layer over it, digest-pinned:

```bash
BASE_IMAGE=ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6 \
TAG=local/vllm-ray-glm53:sm121-v11-dflash2 \
bash cluster/build-image.sh

bash glm/verify-image.sh      # gate: must pass before serving
```

Run both **on each node** — local tags are not registry-backed. `verify-image.sh`
fails if the Ray layer moved any of the base image's pins (which would surface
as wrong numbers, not an error) or if the image does not descend from the pinned
digest.

## Memory budget — why 0.85 is a ceiling

```
weights          181 GiB  ->  90.5 GiB / node at TP=2
budget at 0.85   0.85 x 121.63 GiB = 103.4 GiB / node
headroom         ~12.9 GiB / node for KV + activations + graphs
```

Nemotron runs 1M context with roughly twice this headroom. So:

- `GPU_MEM_UTIL` **0.85 is a ceiling, not a starting point** — `0.90` is
  documented to OOM on this hardware, and the launcher refuses anything above
  0.85 outright.
- KV is fp8 with an explicit 6 GiB budget, not "whatever is left".
- `ENABLE_EAGER` defaults **on**. Graph capture is a memory spike and
  `capture_end` is historically where the Docker cgroup-permission failure first
  surfaced. Only turn it off after 262K is proven stable.

## Bring-up

`PROFILE` is required and has no default.

### Once per node (both Sparks)

There is no shared filesystem, so anything written to disk must exist on **both**
machines. Skipping any of these fails at engine start, not at launch:

```bash
BASE_IMAGE=ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6 \
TAG=local/vllm-ray-glm53:sm121-v11-dflash2 bash cluster/build-image.sh
bash glm/verify-image.sh          # gate: must pass

bash glm/fetch-weights.sh                                     # ~181 GiB target model
MODEL_CKPT=incoai/GLM-5.3-Flash-DFlash2 bash glm/fetch-weights.sh   # drafter, on by default

bash glm/precompile-moe.sh        # ONLY if you intend to use MOE_BACKEND=flashinfer_cutlass
```

`precompile-moe.sh` is not needed for the default `marlin` backend. It is
required before `flashinfer_cutlass`, which otherwise JIT-builds 97 translation
units at KV-profiling time with the weights already resident and gets
earlyoom-killed. See [LADDER.md](LADDER.md) rung 5.

### Every start (three terminals)

```bash
# Node 1 (head)
source glm/cluster-env.sh && make head PROFILE=glm

# Node 2 (worker)
source glm/cluster-env.sh && make worker PROFILE=glm

# Node 1, new terminal
make serve PROFILE=glm
```

That is the whole command — DFlash2 is on by default, so nothing extra is needed
for the fast path. Expect roughly:

| Phase | Duration |
|---|---|
| weight load | ~9 min (rank 0; rank 1 finishes in ~3) |
| `init engine` | ~190 s with DFlash2, ~126 s without |
| **first request** | **slow — this is warmup, not a fault** |

The first generation after startup runs at ~7 tok/s while `mhc_fused_tilelang`
and the xqa decode path compile; subsequent ones settle at ~40. Never benchmark
the first request.

Smoke test:

```bash
source glm/.env
curl -s http://localhost:8000/health && echo " health OK"
RUNS=3 LABEL=check bash glm/bench.sh     # expect ~40 tok/s warm
```

`glm/cluster-env.sh` must be sourced on **both** nodes before bring-up. Ray does
not propagate the head driver's environment to worker ranks across nodes, so a
var set only on the head gives rank 1 a different backend — and that fails as a
collective **hang**, not an error. The launcher refuses to start if the forwarded
vars are missing from the head container.

Requires `glm/.env` (gitignored) — copy `glm/.env.example` and fill in
`HF_TOKEN` and `VLLM_API_KEY`. Neither model repo is gated, so no terms need
accepting; `HF_TOKEN` is only for pull rate limits and the launcher's guard.

## Knobs

| Var | Default | Notes |
|---|---|---|
| `MAX_MODEL_LEN` | `262144` | Checkpoint's native max is higher; raising it needs the forwarded `VLLM_ALLOW_LONG_MAX_MODEL_LEN` |
| `GPU_MEM_UTIL` | `0.85` | Hard ceiling; launcher refuses more |
| `KV_CACHE_MEMORY` | `6442450944` | 6 GiB, fp8. **Auto-traded down to 3 GiB when DFlash2 is on** (i.e. by default) to make room for the 2.34 GB drafter; setting this explicitly opts out. Measured: 3 GiB = 310,292 tokens, 1.18x concurrency at 262K |
| `BLOCK_SIZE` | `2304` | |
| `MAX_NUM_SEQS` | `8` | |
| `MOE_BACKEND` | `marlin` | `flashinfer_cutlass` measured no faster and needs `precompile-moe.sh` first — see LADDER rung 5 |
| `ENABLE_EAGER` | `1` | `0` attempts CUDA graph capture |
| `ENABLE_DFLASH2` | **`1`** | On by default since 2026-09-29: 2.8x decode (40.6 vs 14.4 tok/s). Pulls in the **CC-BY-NC-ND** drafter — set `0` for a licence-clean run |
| `NUM_SPEC_TOKENS` | `7` | Against the drafter's `block_size: 8` |
| `REASONING_PARSER` | `deepseek_r1` | See below |
| `MODEL_CKPT` / `SERVED_NAME` / `PORT` / `TP_SIZE` | — | |

`--tool-call-parser glm47` is hardcoded, not a knob: both sources agree on it and
explicitly warn against `glm` and `glm45`.

`LIMIT_MM` defaults to `{"image":0,"video":0}`. Setting the limits to **zero**
is what makes a run text-only — omitting the flag does not. `--skip-mm-profiling`
only skips the engine's profiling pass; the API server still warms the vision
processor afterwards, which is expensive enough to trigger earlyoom here.

`REASONING_PARSER` was an open question and is now **settled as a negative
result**: no parser in this build surfaces the reasoning. `deepseek_r1` and
`glm47` behave identically — `content` is always correct and never contaminated,
and `reasoning_content` is always `None`. That holds across
`chat_template_kwargs` variants and streaming, so it is a defect in this day-0
build's adapter path, not a misconfiguration. It costs nothing for ordinary use;
the chain-of-thought is simply discarded. To see it, call `/v1/completions` with
the rendered prompt. Do not spend restarts on parser names — see
[LADDER.md](LADDER.md).

## Operational invariants

- **One model at a time.** UMA cannot hold GLM and Nemotron concurrently.
- **Tear down both ranks between runs.** Stale Ray/NCCL processes fight the next
  start. Ctrl-C on the head stops only the head container — `docker stop node-*`
  on the worker by hand.
- **`earlyoom` and `OOMScoreAdjust=-1000` on sshd are prerequisites here**, not
  optional hardening. A `gpt-oss-120b` run once starved this host until sshd was
  unreachable while ICMP still replied, and recovery needed a power cycle. The
  Ray container has no `--memory` cap, so the launcher cannot defend itself.
- **`docker info` must say `cgroupfs`.** Under the systemd driver a
  `daemon-reload` silently revokes the container's GPU device-cgroup permission.
