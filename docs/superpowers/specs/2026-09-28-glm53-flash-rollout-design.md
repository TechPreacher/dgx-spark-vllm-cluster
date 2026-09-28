# GLM-5.3-Flash (NVFP4) rollout on the 2-node DGX Spark cluster

Date: 2026-09-28
Status: approved design, not yet implemented

## Goal

Serve `zai-org/GLM-5.3-Flash` (320B total / 18B active MoE, MIT) as NVFP4 across
both Sparks at tensor-parallel 2, at 262,144-token context, with DFlash2
speculative decoding — added as a third launcher profile alongside the existing
Qwen and Nemotron paths, reusing the Ray/RoCE bring-up this repo already has.

Target: beat Nemotron-3-Super-120B on capability at comparable or better decode
speed. Published figure to validate against is 46.9 tok/s single-stream decode at
74.1% draft acceptance.

## Decisions taken (and what they rule out)

| Decision | Choice | Consequence |
|---|---|---|
| Use case | Personal / research only | DFlash2 drafter (CC-BY-NC-ND-4.0) is permitted. This deployment **cannot** become a commercial product without swapping to MTP. |
| Checkpoint | `LibertAIDAI/GLM-5.3-Flash-NVFP4` (~181 GiB, weight-only NVFP4-A16) | Pairs with DFlash2. Rules out MTP, which only works on `RedHatAI/GLM-5.3-Flash-NVFP4`. |
| Coexistence | Third launcher, `glm/` | Nemotron and Qwen stay. Only one model runs at a time — UMA cannot hold two. |
| Modality | Text-only initially | `--skip-mm-profiling`, no `--limit-mm-per-prompt`. Preserves KV headroom. Vision is a later, separate change. |
| Image strategy | Layer Ray onto the patched GHCR image (approach A) | Inherits 7 day-0 SM121 fixes without owning them. Introduces third-party image trust, mitigated by digest pinning. |

Rejected alternatives:

- **Full local rebuild** of the `v1→v8→v11-dflash2` Dockerfile chain. Auditable
  and trust-free, but hours of ARM build per node and seven patches to own.
  Retained as the documented fallback if the dependency probe (Phase 2) fails.
- **tonyd2wild's Ray-less launcher** (direct rank 0 / rank 1). Closest to the
  proven recipe, but discards this repo's Ray topology, `select_up_dataplane`
  RoCE enumeration and driver preflight, leaving two divergent bring-up paths on
  one cluster.

## Prerequisite: the driver split (gating, blocks everything)

As of 2026-09-27 the two nodes are not serviceable:

| | `pulsar` (head) | `magnetar` (worker) |
|---|---|---|
| Kernel | `7.0.0-1019-nvidia` | `7.0.0-1019-nvidia` |
| `nvidia.ko` | **absent** | loaded |
| Driver userspace | 580.173.02 | 580.178.04 |
| `linux-modules-nvidia-580-open` metapkg | 6.17.0-1032.32 | 7.0.0-1019.19~24.04.2+1 |

`magnetar` was in the phased-update group and rolled onto the 7.0 series
cleanly; `pulsar` took `linux-image-7.0.0-1019-nvidia` but stayed on 6.17
modules and then booted into the gap. `linux-modules-nvidia-580-open-7.0.0-1019-nvidia`
is available in `noble-updates/restricted`, so `scripts/fix_nvidia.sh` recovers
without a reboot and lands `pulsar` on 580.178.04, matching `magnetar`.

Requires an interactive sudo password, so it is run by the operator, not by
tooling.

**Exit criterion:** `scripts/check_nvidia.sh` exits 0 on both nodes *and* both
report the identical driver version. Never bring Ray up across a split-version
pair.

### Detector blind spot to close

During this exact outage `check_nvidia.sh` prints `metapkg lockstep OK`, because
both metapackages agree (at 6.17.0-1032.32). The drift is not between the two
metapackages — it is that a **pinpoint** `linux-image-<ver>-nvidia` package was
installed ahead of the metapackage pair and then booted. The lockstep check
compares the metapackages to each other but never to the running kernel.

Fix: `nvidia_lib.sh` gains a check that the running kernel matches the kernel the
modules metapackage targets, and `check_nvidia.sh` reports drift when it does
not. This is a real gap in the prevention story, not cosmetic — it is why this
outage was not predicted.

## Architecture

### Image layer

`cluster/Dockerfile` is already parameterised on `BASE_IMAGE` and does exactly
one thing: `pip install "ray[default]"`. That is precisely what the patched GHCR
image lacks — it ships "only vLLM + our patches". So GLM needs **no new
Dockerfile**, only a second invocation of the existing build, run on both nodes
(local tags are not registry-backed):

```bash
BASE_IMAGE=ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6 \
TAG=local/vllm-ray-glm53:sm121-v11-dflash2 \
bash cluster/build-image.sh
```

Digest-pinned, never tag-pinned: a third-party tag can move under us.

The day-0 fixes inherited from that image. The recipe advertises seven; six are
documented in enough detail to restate, and the exact inventory is confirmed
against the image during Phase 2 rather than taken on faith:

1. SM121 NoPE MLA — vLLM's SM90 NoPE sparse-MLA backend extended to SM121 via
   the FA2 path (the stock sparse-attention kernel assumes DeepSeek's `pe_dim=64`;
   GLM-5.3-Flash has `qk_rope_head_dim=0`). This is the reason stock vLLM cannot
   run this model on GB10 at all.
2. FlashInfer 0.6.17 produced NaN on SM121 at batch 64–256 rows → 0.6.18 nightly.
3. That nightly silently downgraded NCCL and skewed CUTLASS → re-pinned NCCL
   2.30.7 and CUTLASS 4.6.2.
4. Programmatic Dependent Launch gated off for SM121 (vLLM enabled it for
   capability ≥ 9).
5. Top-k kpool index buffer used `torch.empty` → initialised to `-1` with clamping.
6. FA2 fp8 KV forced `CTA_TILE_KV=32` (a Hopper assumption); capped to 16, which
   is what makes fp8 KV cache usable on GB10.

### Dependency integrity probe

`glm/verify-image.sh` runs **before** any model download. It captures
`torch`, `nccl`, `flashinfer`, `cutlass` versions plus `pip check` inside both
the base image and the layered image, and fails on any delta.

Rationale: those pins are load-bearing for SM121 correctness. If `ray[default]`
drags FlashInfer off 0.6.18, bug 2 returns as silently wrong numerics at certain
batch sizes rather than as a build failure. A cheap check here converts a
subtle, expensive-to-diagnose class of failure into an immediate, named one.

If this probe fails and cannot be resolved by pinning Ray, fall back to
approach B (full local rebuild).

### `glm/` launcher contract

Mirrors `nemotron/`:

- **`glm/cluster-env.sh`** — exports the GLM-specific runtime vars and appends
  their names to `VLLM_FORWARD_VARS`, so the bring-up scripts inject them into
  *both* containers at `docker run` time. Ray does not propagate the head
  driver's `os.environ` to worker ranks across nodes; a rank-1 backend mismatch
  manifests as a collective hang, not an error. Both nodes must source this
  before bring-up.
- **`glm/launch-glm53-flash.sh`** — `docker exec` into the running
  `^node-[0-9]+$` head container and `vllm serve --tensor-parallel-size 2`. Does
  no `docker run` of its own. Refuses to run, printing bring-up instructions, if
  the forwarded vars are not visible inside the head container (copied from the
  Nemotron launcher's guard).
- **`glm/.env`** — `HF_TOKEN`, `VLLM_API_KEY`. Gitignored by the existing
  `**/.env` rule; verify, do not assume.
- **`glm/README.md`** — usage, plus an explicit licence notice: the DFlash2
  drafter is CC-BY-NC-ND-4.0. Research/personal use only; do not redistribute;
  do not bake into any shared image.

**The exact `VLLM_FORWARD_VARS` set is determined during implementation**, by
inspecting the v11-dflash2 image's launcher and entrypoint rather than guessing
from prose recipes. Known-required from the recipes: `NCCL_IB_ADDR_RANGE`
(`10.0.0.0/8`) and `VLLM_ALLOW_LONG_MAX_MODEL_LEN`. Do **not** duplicate
`NCCL_IB_HCA` / `NCCL_SOCKET_IFNAME` / `UCX_NET_DEVICES` — `select_up_dataplane`
in `cluster/lib.sh` already builds those from carrier-up links at launch time,
and a static override would hand NCCL a down HCA after a warm reboot.

### Makefile: generalise, don't multiply targets

Adding `glm-head` / `glm-worker` / `glm` beside the Nemotron trio yields six
near-identical targets and an obvious copy-paste hazard. Instead, parameterise:

```
make head PROFILE=glm
make worker PROFILE=glm
make serve PROFILE=glm
```

`PROFILE` defaults to `nemotron`, so existing invocations keep working
unchanged. The target sources `$(PROFILE)/cluster-env.sh` and takes
`VLLM_IMAGE` from that profile. The `preflight` dependency is unchanged and
still gates on `check_nvidia.sh`.

### Bring-up order

Keep the repo's **head-first** order. tonyd2wild's "worker first, wait ~25 s,
then head" does not transfer: it serves a Ray-less rank rendezvous, whereas this
repo's worker cannot start first because it needs `HEAD_NODE_IP` to join the Ray
cluster.

What does transfer is the intent — do not launch the model until both ranks are
present. The launcher therefore **polls `ray status` until it reports 2 nodes**
(with a timeout) instead of sleeping a fixed interval.

Per-node, before launch:

- `vm.swappiness=0` — UVM livelock defence on GB10 unified memory.
- Page-cache flush.

Note the conflicting advice in the sources: an EXL3-based recipe recommends
`vm.swappiness=180` plus zram. That is a different stack (EXL3/TR3, not NVFP4
weight-only) and is not adopted here. `0` is the NVFP4 path's guidance, and this
cluster's own history — a `gpt-oss-120b` run that starved the host until sshd
was unreachable — makes memory-pressure livelock the failure that has actually
bitten us.

## Memory budget

```
weights          181 GiB  →  90.5 GiB / node
budget at 0.85   0.85 × 121.63 GiB = 103.4 GiB / node
headroom         ~12.9 GiB / node for KV + activations + graphs
```

`--gpu-memory-utilization 0.90` is documented to OOM on this hardware, so 0.85
is a ceiling, not a starting point. Compare Nemotron, which runs 1M context with
roughly twice this headroom — GLM is materially tighter, which is why the
context ladder below exists and why `--enforce-eager` is the default rather than
an opt-in.

Serving flags, converged across the recipes:

```
--tensor-parallel-size 2
--max-model-len 262144
--gpu-memory-utilization 0.85
--kv-cache-dtype fp8
--kv-cache-memory 6442450944
--block-size 2304
--enforce-eager
--tool-call-parser glm47 --enable-auto-tool-choice
--skip-mm-profiling
--speculative-model <dflash2-drafter>  --num-speculative-tokens 7
```

The DFlash2 drafter's exact repo id and `--speculative-model` spelling are
resolved in Phase 2 from the v11-dflash2 image's own launcher, not guessed.

## Recipe conflicts: resolve by probe, not by preference

| Setting | Conflict | Resolution |
|---|---|---|
| `--reasoning-parser` | LibertAI says `deepseek_r1`; MiaAI-Lab says `glm45` | **Probe both** against a thinking prompt and inspect `reasoning_content` vs `content` splitting. A wrong parser does not error — it silently mangles reasoning output, so this must be verified by observation. |
| `vm.swappiness` | `0` (NVFP4 recipe) vs `180` + zram (EXL3 recipe) | Take `0`. Different stack, and local history favours it. |
| `--tool-call-parser` | `glm47`, explicitly not `glm` or `glm45` | Sources agree. Pin it and add a comment so it does not drift. |

## Validation ladder

Climb; do not jump to 262K. Precedent: Nemotron was taken to 1M via a 512K
checkpoint.

| Rung | ctx | util | KV | Spec | Proves |
|---|---|---|---|---|---|
| 1 | 32K | 0.80 | fp8 | off | Weights load; TP2 collectives alive across RoCE |
| 2 | 131K | 0.85 | fp8, 6 GiB | off | KV math holds |
| 3 | 262K | 0.85 | fp8, 6 GiB | off | Target context |
| 4 | 262K | 0.85 | fp8, 6 GiB | DFlash2, 7 | Acceptance + tok/s vs claimed 46.9 / 74.1% |

At every rung record host `MemAvailable` on both nodes and a decode tok/s
number.

**Abort criteria** (any one, on either node): host `MemAvailable` below ~4 GB,
sshd latency degrading, or `dmesg` showing UVM or OOM activity. Recovery from
the documented worst case required a power cycle, so the ladder stops at the
first rung that trips these rather than pressing on.

CUDA graphs stay off (`--enforce-eager`) for rungs 1–3. Capture is a known
memory-spike point and `capture_end` is where the cgroup-permission failure mode
historically first surfaced. Only attempt graph capture after 262K is stable,
and treat it as a separate experiment.

## Guardrails

- `earlyoom` installed and enabled, and `OOMScoreAdjust=-1000` on sshd, on
  **both** Sparks. Previously documented as optional hardening; with roughly half
  Nemotron's headroom these become prerequisites.
- Docker cgroup driver must be `cgroupfs` on both nodes (already true; verify as
  part of preflight). With the systemd driver, any `daemon-reload` silently
  revokes the container's GPU device-cgroup permission mid-run.
- **One model at a time.** UMA cannot hold GLM and Nemotron concurrently. Tear
  down both ranks before relaunching; stale Ray/NCCL processes will fight the
  next start.
- Health checking reuses `cluster/head/ray_inference_health.sh`.

## Documentation

New CLAUDE.md section covering: the GLM path and its launcher contract, the
digest pin and why it is a digest, the parser conflict and how it was resolved,
the one-model-at-a-time invariant, the DFlash2 licence constraint, and the
driver detector blind spot.

## Out of scope

- Vision / video (`--limit-mm-per-prompt`). Separate change once text is stable.
- MTP speculative decoding and the `RedHatAI` checkpoint. Only needed if this
  deployment ever needs to be licence-clean for commercial use.
- CUDA graph capture tuning.
- Retiring Nemotron.
- 4-node / TP4 topologies.
