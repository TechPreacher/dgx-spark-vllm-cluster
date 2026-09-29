# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Bash scripts (no application code) that operate a 2-node NVIDIA DGX Spark (GB10 / SM121) Ray cluster running vLLM inside Docker for distributed LLM inference. The two Sparks are linked by **2 physical 200 GbE ConnectX-7 QSFP ports**. Each physical port is exposed to the OS as two PCIe functions — an `f0` and `f1` half (dual x4-PCIe multi-host, because the GB10 SoC only gives x4 per device) — so `ibdev2netdev` shows 4 interfaces, but the hardware ceiling is **~2×200 = ~400 GbE, not 800**. A **cold boot** brings all four up; a **warm reboot sheds the `f1` half of each port** — the CX7 firmware latches an `insufficient power on the PCIe slot (27W)` state that only a full power cycle (AC removed, PCIe capacitors discharged) clears. The bring-up scripts pin the control plane to the always-up `f0` half and enumerate the data plane from carrier-up links (see below). Models tested: Qwen3-30B-A3B-Thinking-2507-FP8 and Qwen3.5-122B-A10B-FP8 across both nodes. A third launcher (`nemotron/launch-nemotron-120b.sh`) runs NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 across the same 2-node Ray cluster (TP=2) using native NVFP4 on SM121 FP4 tensor cores. A fourth (`glm/launch-glm53-flash.sh`) runs GLM-5.3-Flash (320B-A18B) as weight-only NVFP4 at TP=2 — the only profile that needs its own patched container image, because stock vLLM cannot run that model on GB10 at all.

## Bring-up sequence (must run in order)

`PROFILE` is **required** on `make head` / `make worker` / `make serve` and has **no default**. A bare `make head` fails naming the valid profiles. This is deliberate: the profiles carry different `VLLM_FORWARD_VARS` *and* different container images, so a default would let a bare invocation bring the cluster up on the wrong image with the wrong env — and that does not error, it hangs in an NCCL collective when rank 1 picks a different backend from rank 0. Breaking existing muscle memory was judged cheaper than diagnosing a silent wrong-image bring-up.

1. **Head node (Node 1):** `source <profile>/cluster-env.sh && make head PROFILE=<profile>`
2. **Worker node (Node 2):** `source <profile>/cluster-env.sh && make worker PROFILE=<profile>`
3. **Inject model into running head container** (Node 1, new terminal): `make serve PROFILE=<profile>`

Valid profiles: `nemotron`, `glm`. The Qwen launchers are still invoked directly (`cd qwen && ./launch-qwen-30b.sh`) since the Qwen FP8 path needs no `cluster-env.sh` — bring the cluster up with `PROFILE=nemotron` (its forwarded vars are harmless to Qwen) or run the bring-up scripts by hand.

Each `run_*node_2.sh` script blocks (it `docker run`s with the Ray `--block` command). Closing the terminal stops Ray on that node and tears down the whole cluster. The model-launch script `docker exec`s `vllm serve` into the head container; Ray then schedules TP shard 2 onto Node 2 automatically (`--tensor-parallel-size 2`).

`launch-qwen-*.sh` requires `qwen/.env` containing `VLLM_API_KEY=...` (gitignored — do not commit). They find the head container by matching `^node-[0-9]+$` from `docker ps`.

## The cluster bring-up scripts

- **`run_headnode_2.sh` / `run_workernode_2.sh` (the only bring-up path):** dynamic multi-port data plane. `PRIMARY_IF=enp1s0f0np0` carries Ray control (single IP for `VLLM_HOST_IP` / `MASTER_ADDR`) — pinned to the `f0` half that is up after **both** cold and warm boots (the old `enp1s0f1np1` control IF is gone after a warm reboot). The data-plane vars `DATA_IFS`, `RDMA_HCAS`, `UCX_DEVS` are **built at launch time by `select_up_dataplane` (in `cluster/lib.sh`)** from the CX7 links that currently have carrier — all 4 RoCE HCAs after a cold boot, only the 2 live `f0` HCAs after a warm reboot — so NCCL is never handed a down HCA (which can stall collective init). Those feed `UCX_NET_DEVICES`, `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, `OMPI_MCA_btl_tcp_if_include`, and `NCCL_IB_HCA` (with `NCCL_CROSS_NIC=1`). `TP_SOCKET_IFNAME` is pinned to `PRIMARY_IF` so PyTorch TP setup uses the control IP. Control traffic is coordination-only and does not gate NCCL bandwidth, so full ~2×200G still requires a cold boot (all 4 halves up). Both nodes must be brought up on the matching `10.0.0.x` control subnet (head `10.0.0.1`; worker overrides with `HEAD_NODE_IP`).

Both call `cluster/{head,worker}/run_cluster.sh`. The two copies are kept byte-identical (`--device=/dev/infiniband --cap-add=IPC_LOCK --ulimit memlock=-1:-1` on both, `--shm-size 16g`). If you edit one, copy to the other — RoCE/IB device passthrough must exist on both sides or NCCL silently falls back to TCP and you lose the RoCE data plane.

`run_cluster.sh` positional args: `<image> <head_node_ip> --head|--worker <hf_cache_path> [extra docker args...]`. It extracts `VLLM_HOST_IP` from the extra args, names the container `node-${RANDOM}`, and traps `EXIT` to `docker stop && docker rm` on script exit. Caveat: Ctrl-C on the head terminal stops only the head container; the worker `ray start --block` keeps running attached to a now-dead head — `docker stop node-*` on the worker by hand to clean up.

Optional verbose NCCL logging (off by default): prefix the launch with `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET bash run_*node_2.sh`. Worker head IP override: `HEAD_NODE_IP=10.0.1.x bash run_workernode_2.sh`.

Image: `local/vllm-ray:26.05.post1` — a locally-built tag layered on top of `nvcr.io/nvidia/vllm:26.05.post1-py3` (NGC dropped Ray from the 26.05 vLLM image; `cluster/Dockerfile` adds it back). Build with `bash cluster/build-image.sh` on each node before first use; rebuild after bumping `BASE_IMAGE`. Override per-bring-up with `VLLM_IMAGE=<tag>`. HuggingFace cache is bind-mounted from `~/.cache/huggingface`. Single image for the whole cluster — Qwen and Nemotron share it.

## Model launch — what to know before editing

The Qwen launch scripts encode tuning that is non-obvious:

- **30B Thinking (FP8):** TP=2, ctx 131072, `gpu-memory-utilization 0.70`, `--reasoning-parser deepseek_r1`, `--tool-call-parser hermes`.
- **122B A10B (FP8, Qwen3-Next hybrid Gated DeltaNet + Gated Attention MoE):** TP=2, ctx cut to **65536** (vs native 262k) and util raised to **0.85** because FP8 weights ~125 GB → ~63 GB/node on 128 GB unified memory leaves little headroom for KV+CUDA graphs. Parsers change to `--reasoning-parser qwen3` and `--tool-call-parser qwen3_coder`. `--trust-remote-code` required.
FP8 (not MXFP4) is chosen for the Qwen path to avoid marlin/CUTLASS/FlashInfer-sinks SM121 (GB10) issues. Do not pass `--quantization` on Qwen — vLLM auto-detects from FP8 repos. The `"Config file not found ... GB10.json"` MoE warning at load is harmless (no hand-tuned MoE kernel config for GB10 yet).

## Nemotron-3-Super-120B-A12B-NVFP4 (Ray TP=2 across both Sparks)

`nemotron/launch-nemotron-120b.sh` follows the same pattern as the Qwen launchers: `docker exec` into the running `^node-[0-9]+$` head container and `vllm serve` with `--tensor-parallel-size 2`, letting Ray dispatch shard 2 to the worker over the RoCE data plane. It does NOT do a `docker run` of its own — the Ray cluster must already be up (`run_headnode_2.sh` + `run_workernode_2.sh`).

The model is NVFP4-quantized (native GB10/SM121 FP4 tensor cores) and is a LatentMoE hybrid (Mamba-2 + MoE + Attention). `--mamba-ssm-cache-dtype float16` matters; the reasoning parser is a plugin file (`super_v3_reasoning_parser.py`) the launcher fetches **inside the container** at exec time and stores under `~/.cache/huggingface/` (which is bind-mounted, so it survives Ray container restarts). No host-side bind mount of the parser is needed.

Required env (set in `nemotron/.env`): `HF_TOKEN`, `VLLM_API_KEY`. Unlike the Qwen launchers, `--quantization fp4` and `--moe-backend marlin` are passed explicitly per NVIDIA's DGX Spark example.

**Critical bring-up requirement** that the Qwen path does not have: the four NVFP4 runtime env vars (`VLLM_NVFP4_GEMM_BACKEND=marlin`, `VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm`, `VLLM_USE_FLASHINFER_MOE_FP4=0`, `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`) must be present in each Ray container's env at `docker run` time, **on both nodes**. Ray does not propagate `os.environ` from the head driver to worker ranks across nodes; rank 1 in the worker container therefore inherits only that container's start-time env. If the four vars are missing on the worker, rank 1 picks a different FP4 GEMM kernel / allreduce backend than rank 0 → mismatch, collective hang, or crash at first matmul. The launcher refuses to run if it doesn't see them inside the head container and prints the bring-up instructions.

The mechanism: both `run_headnode_2.sh` and `run_workernode_2.sh` now read a space-separated `VLLM_FORWARD_VARS` from the parent shell and forward each named var into the container as `-e VAR=value`. `nemotron/cluster-env.sh` exports the four NVFP4 vars and sets `VLLM_FORWARD_VARS` to enumerate them. Both nodes must `source nemotron/cluster-env.sh` before running the bring-up script. To add new model-specific runtime flags in the future, append their names to `VLLM_FORWARD_VARS` rather than editing the bring-up scripts.

There is intentionally no `qwen/cluster-env.sh` — the Qwen FP8 path requires no `VLLM_*` runtime-env overrides at container-start time. Bringing the cluster up without sourcing any profile is the correct flow for Qwen. The passthrough loop is a no-op when `VLLM_FORWARD_VARS` is unset.

**Host-stability context:** an earlier `gpt-oss-120b` single-Spark run starved the host of memory so badly that sshd became unreachable while ICMP still replied; recovery required a power cycle. With TP=2, per-node weight memory is roughly halved versus that single-Spark run, but the Ray container has no `--memory` cgroup cap (that's set by `run_cluster.sh` at container create time, not editable after the fact). Defences here are: `--gpu-memory-utilization 0.75` (vs NVIDIA's 0.9), `--max-model-len 1048576` (1M tokens, model maximum; verified stable after a 512k checkpoint with host `MemAvailable` ~18 GB during inference; overridable via `MAX_MODEL_LEN`), and an opt-in `ENABLE_EAGER=1` that skips CUDA graph capture if first-inference memory spikes are a concern. MTP speculative decoding is off by default; enable with `ENABLE_MTP=1`.

Out-of-repo hardening that complements the launcher (apply on **both** Sparks): `OOMScoreAdjust=-1000` on sshd, `earlyoom` installed **and reconfigured** (see below), plus an external laptop-side watchdog that IPMI/PDU-cycles on repeated `/health` failures.

**Installing earlyoom is not enough — stock earlyoom is inert on these hosts.** Its default thresholds are an **AND** across memory and swap: `SIGTERM when mem <= 10% and swap <= 10%`, `SIGKILL when mem <= 5% and swap <= 5%`. Because `vm.swappiness=0` (the UVM-livelock defence) keeps the 16 GB of swap essentially unused, the swap condition is never met and earlyoom never fires under GPU-driven memory pressure. The two defences disarm each other, and `systemctl is-active earlyoom` reporting `active` gives false comfort. Set on both nodes:

```
sudo sed -i 's/^EARLYOOM_ARGS=.*/EARLYOOM_ARGS="-r 3600 -m 4,2 -s 100,100"/' /etc/default/earlyoom
sudo systemctl restart earlyoom
journalctl -u earlyoom --no-pager -n 25 | grep -E 'SIGTERM|SIGKILL' | tail -2
```

`-s 100,100` makes the swap side always true for **both** signals; `-m 4,2` is SIGTERM at ~4.9 GiB and SIGKILL at ~2.5 GiB of 124608 MiB. Give both kill percentages explicitly: with a bare `-m 4 -s 100` earlyoom halves *both*, yielding `SIGKILL when mem <= 2.00% and swap <= 50.00%`, which `swappiness=0` makes unreachable — SIGTERM would work but escalation would be disarmed for the one case it exists for, a process wedged in UVM livelock that ignores SIGTERM. Stock 10% (~12.2 GiB) is too aggressive — Nemotron's own notes record healthy operation at ~18 GB available. Once set, an out-of-memory rung presents as **vLLM being SIGTERMed**, not as a hang: check `journalctl -u earlyoom` before suspecting the model.

**sshd protection on Ubuntu 24.04 here:** `ssh.socket` and `ssh.service` are both active, but the socket runs `Accept=no` and hands the fd to the single `ssh.service` listener — there are no per-connection `ssh@N.service` instances. So one drop-in on `ssh.service` is sufficient; no `ssh@.service` drop-in is needed. Verify with the **listener**, not `pgrep -o sshd` (which returns your own pre-existing session, whose `oom_score_adj` predates the change and reads `0` misleadingly):

```
for p in $(pgrep sshd); do printf '%s %s %s\n' "$p" "$(cat /proc/$p/oom_score_adj)" \
  "$(awk -F/ '{print $NF}' /proc/$p/cgroup | head -1)"; done
```

The process in `ssh.service` must read `-1000`. New logins inherit it across the migration into their `session-N.scope`; existing sessions keep the old value until they reconnect.

## GLM-5.3-Flash-NVFP4 (Ray TP=2 across both Sparks)

`glm/launch-glm53-flash.sh` serves `zai-org/GLM-5.3-Flash` via `LibertAIDAI/GLM-5.3-Flash-NVFP4` — 320B total / 18B active MoE, natively multimodal, hybrid sparse + linear attention with Manifold-Constrained Hyper-Connections (mHC). Target model is MIT. Same pattern as the other launchers: `docker exec` into the running `^node-[0-9]+$` head container, no `docker run` of its own.

**This profile needs its own image, and that is not optional.** Stock vLLM cannot run this model on GB10 at all: it uses NoPE MLA (`qk_rope_head_dim=0`) and the stock sparse-attention kernel assumes DeepSeek's `pe_dim=64`. The serving image is `local/vllm-ray-glm53:sm121-v11-dflash2` — `cluster/Dockerfile`'s existing `ray[default]` layer over a **digest-pinned** patched base:

```
BASE_IMAGE=ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6 \
TAG=local/vllm-ray-glm53:sm121-v11-dflash2 bash cluster/build-image.sh
bash glm/verify-image.sh
```

Digest, never tag: a third-party tag can move under you. `cluster/build-image.sh` stamps the base as a `glm.base.digest` label and `glm/verify-image.sh` gates on it. That base carries the day-0 SM121 fixes (SM90 NoPE sparse-MLA extended to SM121 via FA2, FlashInfer pinned 0.6.18 because 0.6.17 produced NaN at batch 64–256 rows, NCCL 2.30.7, CUTLASS 4.6.2, PDL gated off, FA2 fp8-KV tile capped to 16 which is what makes fp8 KV usable here) but ships **no Ray** — which is exactly the one thing `cluster/Dockerfile` adds. Run the build on each node; local tags are not registry-backed.

`cluster/Dockerfile` also fixes **two** nvrtc gaps in the base image, one for compiling and one for linking. First it links the pip wheel's CUDA headers into `/usr/local/cuda/include`, because FlashInfer's JIT includes `<nvrtc.h>` and the base image ships it only at `dist-packages/nvidia/cu13/include/`; without that the build dies with `fatal error: nvrtc.h: No such file or directory`. Second it creates `/usr/local/cuda/lib64/libnvrtc.so`, the unversioned developer symlink that `ld -lnvrtc` resolves. The image ships only the runtime SONAME `libnvrtc.so.13`, so the **final link fails after all 97 objects have compiled successfully**:
```
/usr/bin/ld: cannot find -lnvrtc: No such file or directory
```
Only nvrtc needs this — `libcudart.so` is present and `libcuda.so` comes from the stubs directory already on the link line. Fixing the headers without the library gets you 97 wasted compiles and a failure at the very last step. Trying CUTLASS is worthwhile because marlin logs *"Your GPU does not have native support for FP4 computation"* and therefore is not using the FP4 tensor cores. (That warning is expected and correct for this weight-only NVFP4-A16 checkpoint; it is not a fallback.)

**The header link is necessary but not sufficient — do not just set `MOE_BACKEND=flashinfer_cutlass` and serve.** The module is 97 nvcc translation units, and vLLM only triggers the build on the first MoE forward, i.e. during KV-cache profiling with 88.63 GiB of weights already resident. That leaves ~10 GiB of host headroom, and a single `cicc` on the worst of these files measures **5284 MiB RSS** — 4.5x the `cudafe++` figure `MAX_JOBS=2` was sized against. earlyoom SIGTERMs the compiler, ninja fails, and the engine dies: observed at object 20 of 97, ~40 minutes into the build on top of a ~9 minute load.

Build it ahead of time instead, with `bash glm/precompile-moe.sh` **on each node** (the JIT cache is per node, like the weights). It runs a throwaway container with no model loaded, so there is ~110 GiB free rather than ~10 GiB, which makes `MAX_JOBS=10` safe and the build far faster than the throttled in-serve attempt. It calls `gen_cutlass_fused_moe_sm120_module(False)` directly — flashinfer's `core.py:568` maps backend `"121"` onto the sm120 module, and both call sites pass only `device_arch`, leaving `use_fast_build` at `False`. Building with the other flag would produce a module vLLM silently never reuses.

The result persists because `run_cluster.sh` bind-mounts `~/.cache/flashinfer`. Before that mount existed the JIT output landed in the container's writable layer and died with the container, so every restart paid the build again. The cache is version-scoped (`0.6.18.dev20260819/cached_ops`), so bumping FlashInfer invalidates it rather than reusing a module built against different headers. **A Ray container started before the mount was added does not have it** — restart the cluster after pulling, or the precompiled module is invisible to the server.

Image IDs will **not** match across the two Sparks — each builds independently and the layers are not bit-reproducible. Compare the **ray version** and the **base digest**, never the image ID.

`glm/verify-image.sh` is a real gate, not a formality: those pins are load-bearing for SM121 *correctness*, so if `ray[default]` moved FlashInfer off 0.6.18 the result is wrong numbers, not a build error. It allows the layer to ADD packages and fails on any version change or removal. It caught a genuine case on first use — an image built before the label existed.

**Memory is the binding constraint, much more than for Nemotron:** 181 GiB of weights → 90.5 GiB/node at TP=2; `0.85 × 121.63` = 103.4 GiB budget; **~12.9 GiB/node** left for KV + activations + graphs, roughly half Nemotron's headroom. Consequences baked into the launcher: `GPU_MEM_UTIL` 0.85 is a **ceiling** and the launcher refuses anything higher (0.90 is documented to OOM); KV is fp8 with an explicit 6 GiB budget rather than "whatever is left"; `--block-size 2304`; `ENABLE_EAGER` defaults **on** (graph capture is a spike, and `capture_end` is where the cgroup-permission failure historically first surfaced).

The checkpoint declares the **multimodal** architecture `Glm5NextForConditionalGeneration` with 43 modules excluded from quantization (BF16 attention, vision tower, shared experts, routers, embeddings, LM head). That is why `--skip-mm-profiling` matters even for a text-only run: the vision tower is in the graph regardless. Vision is deliberately out of scope for now.

`--tool-call-parser glm47` is hardcoded, not a knob — both sources agree and explicitly warn against `glm` and `glm45`. The reasoning parser is a genuine open question (checkpoint card says `deepseek_r1`, a 2-Spark recipe says `glm45`); a wrong one does not error, it silently mis-splits `reasoning_content` from `content`, so it is probed at ladder rung 1. See `glm/LADDER.md`.

**`glm/cluster-env.sh` forwards exactly ONE var, and must not be modelled on nemotron's four.** Two of Nemotron's (`VLLM_NVFP4_GEMM_BACKEND`, `VLLM_USE_FLASHINFER_MOE_FP4`) **do not exist** in this image's vLLM build — it is the patched glm53-flash build (`0.1.dev20051+g487ecf187`), not NGC 26.05, and its env surface differs. `VLLM_ATTENTION_BACKEND` is not env-selectable either, so the SM121 path is chosen by the image's patches rather than by us. Only `VLLM_ALLOW_LONG_MAX_MODEL_LEN` is forwarded, as cheap insurance for raising `MAX_MODEL_LEN` toward the native 1M. `glm/DISCOVERY.md` records every one of these with the command that established it — read it before adding a var.

DFlash2 speculative decoding is **off by default** so a plain run is licence-clean. The drafter `incoai/GLM-5.3-Flash-DFlash2` is **CC-BY-NC-ND-4.0**: research/personal use only, never redistributed, never baked into a shared image. Enable with `ENABLE_DFLASH2=1`. Note the vLLM method string is **`dflash`**, not `dflash2` — the "2" is in the drafter's architecture (`DFlash2DraftModel`, which resolves to the Qwen3 DFlash2 class because the drafter is Qwen3-shaped). If this deployment ever needs to be licence-clean *and* fast, the alternative is MTP, which requires the `RedHatAI/GLM-5.3-Flash-NVFP4` checkpoint instead.

### Serving flags this model actually needs (all learned the hard way)

Every one of these cost a failed bring-up, several of them ~9 minutes into a load. They are in `glm/launch-glm53-flash.sh`; this is why.

| Flag | Why |
|---|---|
| `--distributed-executor-backend ray` | This build defaults to `mp` (`config/parallel.py:917`) and does **not** infer Ray from a live cluster the way NGC 26.05 does. Without it: *"World size (2) is larger than the number of available GPUs (1)"* at config time. |
| `--kv-cache-memory-bytes` | `--kv-cache-memory` is **not a registered flag**; it resolves only through argparse prefix-abbreviation and breaks the moment another `--kv-cache-memory*` option appears. |
| `--moe-backend marlin` | The auto choice (`FLASHINFER_CUTLASS`) JIT-builds `fused_moe_120` at profiling time, which cannot succeed there — it needs `nvrtc.h` (shipped only inside the pip wheel) and ~5.3 GiB per compiler process against ~10 GiB of headroom. marlin is prebuilt and needs no JIT. Switch to `flashinfer_cutlass` only after `glm/precompile-moe.sh` has run on both nodes; see the image note below. |
| `--limit-mm-per-prompt {"image":0,"video":0}` | **`--skip-mm-profiling` alone does NOT give a text-only run.** It skips the engine's profiling pass; the API server still warms the vision processor afterwards (`renderers/base.py`), which took 51 s + 24 s and got rank 0 OOM-killed *after* `Application startup complete`. The warmup gate is `mm_limits = {k: v for k, v in allowed_mm_limits.items() if v > 0}`, so the limits must be **set to zero**, not omitted. Confirmed working when the log says `running in text-only mode`. |
| `--kernel-config` disabling autotune + warmups | FlashInfer autotune (~53 s) plus repeated TileLang compiles spike host memory *after* the KV cache is already reserved. earlyoom SIGTERMed rank 0 there. Kernels still compile lazily on first use. |
| `MAX_JOBS=2` (forwarded) | `MAX_JOBS` defaults to the **CPU count** (20) and drives ninja's parallel workers in FlashInfer's JIT; each spawns a `cudafe++` at ~1.17 GiB, so the default fans out to ~23 GiB of compiler memory on top of resident weights. |

**A repo-id `--model` does not work.** `Glm5NextProcessor.from_pretrained` does a raw `open(os.path.join(model_path, "processor_config.json"))` (`transformers_utils/processors/glm5next.py:853`) instead of resolving through the Hub, so it only accepts a local directory. The launcher resolves the repo id via `snapshot_download` and passes the snapshot path, keeping `--served-model-name` so clients are unaffected.

**`GPU_MEM_UTIL` is inert while `kv_cache_memory_bytes` is set.** vLLM says so in the log: *"reserved 6.0 GiB ... and skipped memory profiling. This does not respect the gpu_memory_utilization config."* The launcher still refuses values above 0.85, which matters only if the KV budget is ever unset.

### Memory is the binding constraint, and pulsar is the binding node

Measured at 262K, text-only, model resident:

| | pulsar | magnetar |
|---|---|---|
| idle | **6.6 GB** | 11.1 GB |
| during a 60K-token prefill | **6.1 GB** | 10.6 GB |
| earlyoom SIGTERM at | ~4.9 GB | ~4.9 GB |

**pulsar runs ~4.5 GB tighter than magnetar, every time.** Working margin there is 1.2–1.7 GB. KV is *not* the limit — 6 GiB buys 925,447 tokens (3.53x concurrency at 262K) — so `KV_CACHE_MEMORY` is the lever with the most slack whenever headroom is needed. Enabling DFlash2 requires it: the drafter is 2.34 GB, more than the whole margin, so the launcher trades KV 6 GiB → 3 GiB automatically when `ENABLE_DFLASH2=1`.

Every memory failure in this deployment presented as something else — a Ray `SYSTEM_ERROR`, a "connection error code 2", a worker dying with no message. **`journalctl -u earlyoom` is the first thing to check**, not the last; it names the process and the threshold every time.

### The reasoning parser drops the chain-of-thought (open defect)

The chat template ends the prompt with `<|assistant|><think>` (`chat_template.jinja:256`), so `<think>` is in the **prompt** and the model emits reasoning then `</think>`. `content` is always correct and never contaminated — but `reasoning_content` is `None` with **every** parser tried (`deepseek_r1`, `glm47`), with `chat_template_kwargs` `{"thinking":true}` and `{"enable_thinking":true}`, and in streaming (0 reasoning deltas).

The state machine is working as designed — `parser/glm47_moe.py:125` sets `initial_state=ParserState.REASONING if thinking`, which is exactly why content stays clean — but the REASONING events never reach `reasoning_content` in either aggregation path. This looks like a defect in this day-0 build. It costs nothing for ordinary use; the chain-of-thought is simply discarded. To see it, call `/v1/completions` with the rendered prompt.

Note the template's kwargs are **`reasoning_effort`** (`low`/`high`, default `max`) and **`clear_thinking`** — *not* the `enable_thinking` the published recipes mention, which this template ignores entirely.

**Short outputs look like a parser failure and are not.** With a small `max_tokens` the model never closes `</think>`, so everything lands in the discarded reasoning and both `content` and `reasoning_content` come back `None`. Give it ≥256 tokens before concluding anything.

**The HuggingFace cache is per node, and Ray TP needs the checkpoint on EVERY node.** `run_cluster.sh` bind-mounts `~/.cache/huggingface` from each host separately — there is no shared filesystem — and each rank loads its shard from its own node's disk. A checkpoint present only on the head fails ~30 s into engine init as a Ray traceback that names the path but not the reason: `ray::RayWorkerProc.initialize_worker() (ip=10.0.0.2) RuntimeError: Cannot find any model weights with '/root/.cache/.../snapshots/...'`, while rank 0 happily logs `Checkpoint size: 181.30 GiB`. So GLM needs ~181 GiB on **both** Sparks, ~362 GiB total.

Populate with `bash glm/fetch-weights.sh` on each node (resumable, ~30 min each, both can run in parallel). It downloads from inside a throwaway container **on purpose**: the cache directories are created by the Ray container as root, so the host user cannot write into them — a host-side `hf download` or an `rsync` from the head fails on permissions even though the blobs themselves are world-readable. Writing from a container keeps ownership consistent with the serving path and needs no sudo.

`glm/launch-glm53-flash.sh` preflights this by probing every Ray node through `NodeAffinitySchedulingStrategy`, so it reports `magnetar dir=True safetensors=0 MISSING` before serving rather than failing inside the engine. Bypass with `SKIP_WEIGHT_CHECK=1`.

**One model at a time** — UMA cannot hold GLM and Nemotron concurrently. Tear down both ranks before relaunching; stale Ray/NCCL processes fight the next start.

`earlyoom` and `OOMScoreAdjust=-1000` on sshd are **prerequisites** for this profile, not the optional hardening they are for Nemotron: half the headroom, and the `gpt-oss-120b` precedent needed a power cycle. Also `vm.swappiness=0` on both nodes (UVM-livelock defence on GB10 unified memory; one EXL3 recipe says `180` + zram, but that is a different stack and is not adopted here).

## Docker GPU cgroup gotcha (applies to every model, both nodes)

Docker must run with the **cgroupfs** cgroup driver on both Sparks — set once via `/etc/docker/daemon.json` = `{ "exec-opts": ["native.cgroupdriver=cgroupfs"] }`, then `sudo systemctl restart docker`. Verify with `docker info | grep -i "Cgroup Driver"` (must say `cgroupfs`). With the default **systemd** driver, any `systemctl daemon-reload` while a Ray container is running — including the automatic ones `snapd` fires to refresh snap-confine AppArmor profiles — makes systemd re-derive the container scope's device cgroup and **silently drop the nvidia-container-toolkit-injected `/dev/nvidia*` devices** (toolkit runs with `no-cgroups=false`). The device *nodes* stay mounted (`ls /dev/nvidia*` inside the container still works) but the container loses cgroup *permission* to use them. Symptom: a running model dies mid-flight with `CUDA error: operation not permitted` / `cudaErrorNotPermitted` (often first surfacing at CUDA-graph `capture_end`), and any subsequent launch fails earlier with `Failed to initialize NVML: Unknown Error` and `current platform None does not support ray`. Recovery once bitten: recreate the affected container (`make worker` / `make head`). `daemon.json` is read at every Docker start, so the fix survives reboots. This is unrelated to the warm-reboot link degradation above — it's a container-cgroup issue, not a fabric or GPU-hardware fault.

## NVIDIA driver / kernel-module lockstep (applies to every model, both nodes)

The Sparks have **no DKMS** — nothing rebuilds NVIDIA modules at boot. Kernel modules come only from prebuilt `linux-modules-nvidia-<branch>-<kernel>` packages. The kernel (src: `linux-nvidia`) and the driver (src: `nvidia-graphics-drivers-<branch>`) are **separate source packages on independent phased-update schedules**, and Ubuntu decides phasing **per-machine** (deterministic on machine-id). So one `apt upgrade` can pull a new kernel while holding the driver back — on one Spark but not the other. Reboot into that gap and the running kernel has no `nvidia.ko` at all.

Symptom: `make head` dies in the container prestart hook with `nvidia-container-cli: initialization error: nvml error: driver not loaded`. On the host, `lsmod | grep nvidia` is empty, `/dev/nvidia*` is absent, and `nvidia-smi` reports it "couldn't communicate with the NVIDIA driver". This is **not** the cgroup issue above — that one revokes GPU access from a *running* container with the driver loaded fine; this is the host driver being absent entirely.

Seen 2026-08-20: `apt upgrade -y` took `linux-image-nvidia-hwe-24.04` 6.17.0-1026.26 → 6.17.0-1029.29 but left `linux-modules-nvidia-580-open-nvidia-hwe-24.04` at 6.17.0-1026.26; reboot 14 minutes later ⇒ no GPU on `pulsar`. `magnetar`, same command same day, was in the phase group and came up fine — which is exactly why driver version must be compared across both nodes, not assumed.

Seen again 2026-09-27 on `pulsar`, and this instance had a **different shape** worth recognising. Both metapackages agreed with each other at 6.17.0-1032.32; what had happened was that a **pinpoint** `linux-image-7.0.0-1019-nvidia` was installed *ahead* of the metapackage pair and then booted. So the running kernel was 7.0.0-1019 with no `nvidia.ko`, while `check_nvidia.sh` reported `metapkg lockstep OK` throughout the outage — the lockstep test compares the two metapackages to each other and never to the running kernel. `magnetar` had rolled cleanly onto the 7.0 series and was healthy on 580.178.04, so the pair was also split across the fabric.

`scripts/check_nvidia.sh` now carries a `kernel covered` row for exactly this: it compares the running kernel's ABI against the kernel the modules metapackage targets, using the pure helpers `kernel_abi_from_release` / `kernel_abi_from_pkg_version` in `scripts/nvidia_lib.sh` (unit-tested by `scripts/test_nvidia_lib.sh`, `make test`). It **warns** rather than fails: when a pinpoint modules package does cover the running kernel the GPU is fine today and the exposure is the next upgrade; when it does not, `nvidia.ko MISSING` has already failed and this row supplies the cause. Recovery was the standard `scripts/fix_nvidia.sh` — it installed the modules for the running kernel, moved the userspace 580.173.02 → 580.178.04, re-armed both metapackages onto the 7.0 series, and rebuilt 6.17.0-1032's modules against the new driver so the fallback kernel stayed bootable. No reboot needed.

Recovery needs no reboot (the modules target the already-running kernel): `bash scripts/fix_nvidia.sh`, which installs the **metapackage** (not the pinpoint `...-<kernel>` package — the metapackage is what drifted, so upgrading it re-arms lockstep for the next kernel) and `modprobe`s. Expect it to move the whole NVIDIA userspace to a new driver version; that is correct, since the modules package hard-depends on a matching `nvidia-kernel-common-<branch>`, so module and userspace advance together by construction. It also rebuilds the *previous* kernel's modules against the new driver, keeping the old kernel bootable as a fallback.

**Both Sparks must end up on the identical driver version** before bringing Ray up — don't leave a split-version pair across the RoCE fabric.

Prevention is `scripts/preboot_check.sh`, run on each node before any reboot: it checks the kernel GRUB will boot **next**, not the running one. An apt-level mitigation (`APT::Get::Always-Include-Phased-Updates "true"`) would narrow the race but cannot close it — the archive can publish a kernel before its matching modules package exists — so the pre-reboot check stays the real defence.

## Health / monitoring

- `cluster/head/ray_inference_health.sh` — `ray status` in container, `curl :8000/health`, `nvidia-smi` on host + in container. Exits non-zero if no `node-*` container is running.
- `scripts/check_nvidia.sh` (`make check-nvidia`) — host NVIDIA driver health for **this** node; run it on each Spark. Exit codes are split so bring-up can gate on real breakage only: `0` healthy, `1` driver unusable now (blocks `make head`/`make worker`), `3` driver fine but metapackages drifted (next-reboot risk — warns, does not block), `2` the check itself could not run.
- `scripts/preboot_check.sh` (`make preboot-check`) — run **before rebooting** a node: verifies the next-boot kernel has NVIDIA modules. `SAFE` / `UNSAFE TO REBOOT`.
- `scripts/fix_nvidia.sh` (`make fix-nvidia`) — installs the matching modules metapackage and loads it. Shows an `apt-get -s` preview and prompts; `FIX_YES=1` to skip the prompt.
- `scripts/test_nvidia_lib.sh` (`make test`) — unit tests for the pure ABI-extraction helpers in `nvidia_lib.sh`. No root, no host mutation.
- `glm/verify-image.sh` — gates the GLM serving image: asserts it descends from the pinned base digest and that the `ray[default]` layer changed or removed **no** package the base pinned. Run on each node after building. Those pins are why SM121 works, and moving one yields wrong numbers rather than an error.
- `scripts/nvidia_lib.sh` — shared helpers for the three above (driver-branch detection, next-boot kernel, `nvidia.ko` probe). Sourced, not executed. Branch is detected from installed packages, never hardcoded, so a 580 → 590 bump needs no edit.
- `make head` / `make worker` run `check_nvidia.sh` as a preflight and abort on exit 1/2, turning the opaque `nvml error: driver not loaded` into a named cause plus the fix command. Bypass with `SKIP_PREFLIGHT=1`. Both also require `PROFILE` first (see the bring-up section) — the profile check runs before the preflight, so a missing `PROFILE` fails without touching the driver.

## Things that look risky and aren't (and vice-versa)

- The `Warning: VLLM_HOST_IP differs from head_node_ip` branch in `run_cluster.sh` resolves by trusting `VLLM_HOST_IP` — intentional.
- `RAY_memory_monitor_refresh_ms=0` disables Ray's OOM killer; deliberate for vLLM workloads.
- `TP_SOCKET_IFNAME=$PRIMARY_IF` (not `$DATA_IFS`) is intentional — PyTorch TP rendezvous needs a single IP; data plane is for NCCL/UCX.
- `UCX_NET_DEVICES` uses RDMA device names with `:1` port suffix (e.g. `rocep1s0f0:1`), not netdev names — UCX needs the RDMA device path to use the RoCE transport rather than falling back to TCP.
- `CONTAINER_NAME="node-$(date +%s)$$"` keeps the `^node-[0-9]+$` shape that all the launcher / health scripts grep for, while avoiding the `$RANDOM` (15-bit) collision risk across rapid reruns.
