# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Bash scripts (no application code) that operate a 2-node NVIDIA DGX Spark (GB10 / SM121) Ray cluster running vLLM inside Docker for distributed LLM inference. The two Sparks are linked by **2 physical 200 GbE ConnectX-7 QSFP ports**. Each physical port is exposed to the OS as two PCIe functions — an `f0` and `f1` half (dual x4-PCIe multi-host, because the GB10 SoC only gives x4 per device) — so `ibdev2netdev` shows 4 interfaces, but the hardware ceiling is **~2×200 = ~400 GbE, not 800**. A **cold boot** brings all four up; a **warm reboot sheds the `f1` half of each port** — the CX7 firmware latches an `insufficient power on the PCIe slot (27W)` state that only a full power cycle (AC removed, PCIe capacitors discharged) clears. The bring-up scripts pin the control plane to the always-up `f0` half and enumerate the data plane from carrier-up links (see below). Models tested: Qwen3-30B-A3B-Thinking-2507-FP8 and Qwen3.5-122B-A10B-FP8 across both nodes. A third launcher (`nemotron/launch-nemotron-120b.sh`) runs NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 across the same 2-node Ray cluster (TP=2) using native NVFP4 on SM121 FP4 tensor cores.

## Bring-up sequence (must run in order)

1. **Head node (Node 1, `10.0.1.3`):** `cd cluster/head && bash run_headnode_2.sh`
2. **Worker node (Node 2):** `cd cluster/worker && bash run_workernode_2.sh`
3. **Inject model into running head container:** `cd qwen && ./launch-qwen-30b.sh` (or `./launch-qwen-122b.sh`)

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

Out-of-repo hardening that complements the launcher (apply on **both** Sparks): `OOMScoreAdjust=-1000` on sshd via `systemctl edit ssh`, `earlyoom` package installed and enabled, plus an external laptop-side watchdog that IPMI/PDU-cycles on repeated `/health` failures.

## Docker GPU cgroup gotcha (applies to every model, both nodes)

Docker must run with the **cgroupfs** cgroup driver on both Sparks — set once via `/etc/docker/daemon.json` = `{ "exec-opts": ["native.cgroupdriver=cgroupfs"] }`, then `sudo systemctl restart docker`. Verify with `docker info | grep -i "Cgroup Driver"` (must say `cgroupfs`). With the default **systemd** driver, any `systemctl daemon-reload` while a Ray container is running — including the automatic ones `snapd` fires to refresh snap-confine AppArmor profiles — makes systemd re-derive the container scope's device cgroup and **silently drop the nvidia-container-toolkit-injected `/dev/nvidia*` devices** (toolkit runs with `no-cgroups=false`). The device *nodes* stay mounted (`ls /dev/nvidia*` inside the container still works) but the container loses cgroup *permission* to use them. Symptom: a running model dies mid-flight with `CUDA error: operation not permitted` / `cudaErrorNotPermitted` (often first surfacing at CUDA-graph `capture_end`), and any subsequent launch fails earlier with `Failed to initialize NVML: Unknown Error` and `current platform None does not support ray`. Recovery once bitten: recreate the affected container (`make worker` / `make head`). `daemon.json` is read at every Docker start, so the fix survives reboots. This is unrelated to the warm-reboot link degradation above — it's a container-cgroup issue, not a fabric or GPU-hardware fault.

## NVIDIA driver / kernel-module lockstep (applies to every model, both nodes)

The Sparks have **no DKMS** — nothing rebuilds NVIDIA modules at boot. Kernel modules come only from prebuilt `linux-modules-nvidia-<branch>-<kernel>` packages. The kernel (src: `linux-nvidia`) and the driver (src: `nvidia-graphics-drivers-<branch>`) are **separate source packages on independent phased-update schedules**, and Ubuntu decides phasing **per-machine** (deterministic on machine-id). So one `apt upgrade` can pull a new kernel while holding the driver back — on one Spark but not the other. Reboot into that gap and the running kernel has no `nvidia.ko` at all.

Symptom: `make head` dies in the container prestart hook with `nvidia-container-cli: initialization error: nvml error: driver not loaded`. On the host, `lsmod | grep nvidia` is empty, `/dev/nvidia*` is absent, and `nvidia-smi` reports it "couldn't communicate with the NVIDIA driver". This is **not** the cgroup issue above — that one revokes GPU access from a *running* container with the driver loaded fine; this is the host driver being absent entirely.

Seen 2026-08-20: `apt upgrade -y` took `linux-image-nvidia-hwe-24.04` 6.17.0-1026.26 → 6.17.0-1029.29 but left `linux-modules-nvidia-580-open-nvidia-hwe-24.04` at 6.17.0-1026.26; reboot 14 minutes later ⇒ no GPU on `pulsar`. `magnetar`, same command same day, was in the phase group and came up fine — which is exactly why driver version must be compared across both nodes, not assumed.

Recovery needs no reboot (the modules target the already-running kernel): `bash scripts/fix_nvidia.sh`, which installs the **metapackage** (not the pinpoint `...-<kernel>` package — the metapackage is what drifted, so upgrading it re-arms lockstep for the next kernel) and `modprobe`s. Expect it to move the whole NVIDIA userspace to a new driver version; that is correct, since the modules package hard-depends on a matching `nvidia-kernel-common-<branch>`, so module and userspace advance together by construction. It also rebuilds the *previous* kernel's modules against the new driver, keeping the old kernel bootable as a fallback.

**Both Sparks must end up on the identical driver version** before bringing Ray up — don't leave a split-version pair across the RoCE fabric.

Prevention is `scripts/preboot_check.sh`, run on each node before any reboot: it checks the kernel GRUB will boot **next**, not the running one. An apt-level mitigation (`APT::Get::Always-Include-Phased-Updates "true"`) would narrow the race but cannot close it — the archive can publish a kernel before its matching modules package exists — so the pre-reboot check stays the real defence.

## Health / monitoring

- `cluster/head/ray_inference_health.sh` — `ray status` in container, `curl :8000/health`, `nvidia-smi` on host + in container. Exits non-zero if no `node-*` container is running.
- `scripts/check_nvidia.sh` (`make check-nvidia`) — host NVIDIA driver health for **this** node; run it on each Spark. Exit codes are split so bring-up can gate on real breakage only: `0` healthy, `1` driver unusable now (blocks `make head`/`make worker`), `3` driver fine but metapackages drifted (next-reboot risk — warns, does not block), `2` the check itself could not run.
- `scripts/preboot_check.sh` (`make preboot-check`) — run **before rebooting** a node: verifies the next-boot kernel has NVIDIA modules. `SAFE` / `UNSAFE TO REBOOT`.
- `scripts/fix_nvidia.sh` (`make fix-nvidia`) — installs the matching modules metapackage and loads it. Shows an `apt-get -s` preview and prompts; `FIX_YES=1` to skip the prompt.
- `scripts/nvidia_lib.sh` — shared helpers for the three above (driver-branch detection, next-boot kernel, `nvidia.ko` probe). Sourced, not executed. Branch is detected from installed packages, never hardcoded, so a 580 → 590 bump needs no edit.
- `make head` / `make worker` run `check_nvidia.sh` as a preflight and abort on exit 1/2, turning the opaque `nvml error: driver not loaded` into a named cause plus the fix command. Bypass with `SKIP_PREFLIGHT=1`.

## Things that look risky and aren't (and vice-versa)

- The `Warning: VLLM_HOST_IP differs from head_node_ip` branch in `run_cluster.sh` resolves by trusting `VLLM_HOST_IP` — intentional.
- `RAY_memory_monitor_refresh_ms=0` disables Ray's OOM killer; deliberate for vLLM workloads.
- `TP_SOCKET_IFNAME=$PRIMARY_IF` (not `$DATA_IFS`) is intentional — PyTorch TP rendezvous needs a single IP; data plane is for NCCL/UCX.
- `UCX_NET_DEVICES` uses RDMA device names with `:1` port suffix (e.g. `rocep1s0f0:1`), not netdev names — UCX needs the RDMA device path to use the RoCE transport rather than falling back to TCP.
- `CONTAINER_NAME="node-$(date +%s)$$"` keeps the `^node-[0-9]+$` shape that all the launcher / health scripts grep for, while avoiding the `$RANDOM` (15-bit) collision risk across rapid reruns.
