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
use only, no redistribution, and never baked into a shared image. It is **off by
default** (`ENABLE_DFLASH2=0`); a default run is licence-clean. If this
deployment ever needs to serve commercial traffic, the drafter must go — the
licence-clean alternative is MTP, which requires the
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

```bash
# Node 1 (head)
source glm/cluster-env.sh && make head PROFILE=glm

# Node 2 (worker)
source glm/cluster-env.sh && make worker PROFILE=glm

# Node 1, new terminal
make serve PROFILE=glm
```

`glm/cluster-env.sh` must be sourced on **both** nodes before bring-up. Ray does
not propagate the head driver's environment to worker ranks across nodes, so a
var set only on the head gives rank 1 a different backend — and that fails as a
collective **hang**, not an error. The launcher refuses to start if the forwarded
vars are missing from the head container.

Requires `glm/.env` (gitignored) — copy `glm/.env.example` and fill in
`HF_TOKEN` and `VLLM_API_KEY`.

## Knobs

| Var | Default | Notes |
|---|---|---|
| `MAX_MODEL_LEN` | `262144` | Checkpoint's native max is higher; raising it needs the forwarded `VLLM_ALLOW_LONG_MAX_MODEL_LEN` |
| `GPU_MEM_UTIL` | `0.85` | Hard ceiling; launcher refuses more |
| `KV_CACHE_MEMORY` | `6442450944` | 6 GiB, fp8 |
| `BLOCK_SIZE` | `2304` | |
| `MAX_NUM_SEQS` | `8` | |
| `ENABLE_EAGER` | `1` | `0` attempts CUDA graph capture |
| `ENABLE_DFLASH2` | `0` | `1` enables the CC-BY-NC-ND drafter |
| `NUM_SPEC_TOKENS` | `7` | Against the drafter's `block_size: 8` |
| `REASONING_PARSER` | `deepseek_r1` | See below |
| `MODEL_CKPT` / `SERVED_NAME` / `PORT` / `TP_SIZE` | — | |

`--tool-call-parser glm47` is hardcoded, not a knob: both sources agree on it and
explicitly warn against `glm` and `glm45`.

`REASONING_PARSER` is a genuine open question — the checkpoint card says
`deepseek_r1`, a 2-Spark recipe says `glm45`. A wrong parser does **not** error;
it silently mis-splits `reasoning_content` from `content`. See
[LADDER.md](LADDER.md) for the probe result.

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
