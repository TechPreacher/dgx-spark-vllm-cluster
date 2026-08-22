# dgx-spark-vllm-cluster

Operator scripts for a 2-node NVIDIA DGX Spark cluster running distributed LLM inference via Ray + vLLM in Docker.

## Hardware

- 2× NVIDIA DGX Spark (GB10 / SM121, 128 GB unified memory each)
- 2 physical 200 GbE ConnectX-7 QSFP ports per node, each exposed as two PCIe `f0`/`f1` halves (4 OS interfaces, ~2×200 = ~400 GbE ceiling; RoCE). Cold boot brings all 4 up; a warm reboot sheds the `f1` half of each port (PCIe 27 W power latch — cold-boot to recover)
- Control plane pinned to the always-up `f0` half (`enp1s0f0np0`) on a `/24` carrying Ray + TP rendezvous; the data plane is enumerated from carrier-up links at bring-up (`select_up_dataplane`)
- Head node primary IP: `10.0.1.3` (override via `HEAD_NODE_IP` env)

## Prerequisites

- Docker with NVIDIA runtime
- HuggingFace cache at `~/.cache/huggingface` (writable, shared via bind mount)
- `qwen/.env` with `VLLM_API_KEY=…` (not committed — see `.gitignore`)
- Optional pre-stage: `hf download Qwen/Qwen3-30B-A3B-Thinking-2507-FP8`

## Layout

```
cluster/
  Dockerfile                 # FROM nvcr.io/nvidia/vllm:26.05.post1-py3 + ray[default]
  build-image.sh             # run once per node before first bring-up
  lib.sh                     # shared: load_env, find_ray_container (sourced by launchers + health)
  head/
    run_headnode_2.sh        # cluster bring-up (dynamic multi-port data plane)
    run_cluster.sh           # generic Ray+Docker launcher (head/worker)
    ray_inference_health.sh  # ray status + /health + nvidia-smi
  worker/
    run_workernode_2.sh      # cluster bring-up (dynamic multi-port data plane)
    run_cluster.sh           # byte-identical to head/run_cluster.sh
qwen/
  launch-qwen-30b.sh         # Qwen3-30B-A3B-Thinking-2507-FP8 (TP=2, 131k ctx)
  launch-qwen-122b.sh        # Qwen3.5-122B-A10B-FP8 (TP=2, 65k ctx, Qwen3-Next hybrid)
  .env.example               # VLLM_API_KEY + HF_TOKEN template
nemotron/
  launch-nemotron-120b.sh    # NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 (Ray TP=2, NVFP4)
  cluster-env.sh             # NVFP4 runtime env -- MUST be sourced before cluster bring-up
  .env.example               # VLLM_API_KEY + HF_TOKEN template
scripts/                     # per-node host checks; all operate on the node they run on
  check_cluster.sh           # ibdev2netdev (netdev <-> RoCE HCA mapping)
  check_docker.sh            # cgroup driver + GPU visible inside the Ray container
  fix_docker.sh              # set cgroupfs driver, restart docker
  nvidia_lib.sh              # shared: driver-branch detect, next-boot kernel, nvidia.ko probe
  check_nvidia.sh            # host NVIDIA driver health (make check-nvidia)
  preboot_check.sh           # before rebooting: will this node come back with a GPU?
  fix_nvidia.sh              # install matching NVIDIA modules metapackage + load it
CLAUDE.md                    # operator notes for Claude Code
```

## One-time per-node image build

NGC dropped Ray from the `nvcr.io/nvidia/vllm:26.05*` images. `cluster/Dockerfile` layers `ray[default]` back on top. Build the resulting `local/vllm-ray:26.05.post1` tag on **both** nodes once before the first bring-up (and after any base-image bump):

```bash
bash cluster/build-image.sh    # on Node 1
bash cluster/build-image.sh    # on Node 2
```

Override the base or tag via env: `BASE_IMAGE=nvcr.io/nvidia/vllm:26.06-py3 TAG=local/vllm-ray:26.06 bash cluster/build-image.sh`.

## Bring-up (order matters)

```bash
# 1. Head node (Node 1)
cd cluster/head
bash run_headnode_2.sh

# 2. Worker node (Node 2)
cd cluster/worker
bash run_workernode_2.sh

# 3. Inject model into running head container (back on Node 1, new terminal)
cd qwen
./launch-qwen-30b.sh        # or ./launch-qwen-122b.sh
```

Each cluster script blocks. Closing the terminal stops Ray on that node and tears down the cluster. The model-launch script uses `docker exec` against the head container (matched via `^node-[0-9]+$` from `docker ps`) and runs `vllm serve`. Ray dispatches TP shard 2 to Node 2 automatically.

### Makefile shortcuts

A top-level `Makefile` wraps the three bring-up commands. It sources `nemotron/cluster-env.sh` on both nodes so the cluster comes up Nemotron-ready (no-op for the Qwen path — `VLLM_FORWARD_VARS` passthrough does nothing when its target vars are unset at vLLM-serve time on the Qwen launchers).

```bash
# Node 1 (head)
make head

# Node 2 (worker)
make worker

# Node 1, new terminal, after both nodes are up
make nemotron
```

Host-check targets, each operating on the node they are run from:

```bash
make check-nvidia    # is the NVIDIA driver healthy on this node?
make preboot-check   # before rebooting this node: will it come back with a GPU?
make fix-nvidia      # install NVIDIA modules matching this node's kernel
```

`make head` and `make worker` run `check-nvidia` as a preflight and refuse to start the cluster if the host driver is unusable — otherwise the failure surfaces only as an opaque `nvml error: driver not loaded` from Docker. Bypass with `SKIP_PREFLIGHT=1`.

Run `make help` to list targets. The Qwen launchers still live at `qwen/launch-qwen-*.sh` and are not wrapped by the Makefile.

## Verify

```bash
# In head terminal session:
bash cluster/head/ray_inference_health.sh

# Smoke test from another machine:
curl http://<head-ip>:8000/health
curl http://<head-ip>:8000/v1/chat/completions \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3_30b_thinking","messages":[{"role":"user","content":"12*17"}],"max_tokens":200}'
```

## Models

| Script | Model | TP | ctx | mem-util | Notes |
|---|---|---|---|---|---|
| `qwen/launch-qwen-30b.sh` | Qwen3-30B-A3B-Thinking-2507-FP8 | 2 | 131072 | 0.70 | `deepseek_r1` reasoning, `hermes` tools |
| `qwen/launch-qwen-122b.sh` | Qwen3.5-122B-A10B-FP8 | 2 | 65536 | 0.85 | Qwen3-Next hybrid MoE; ctx cut from 262k to fit KV+CUDA graphs |
| `nemotron/launch-nemotron-120b.sh` | NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 | 2 | 1048576 | 0.75 | Ray TP=2 across both Sparks; NVFP4 native FP4 on SM121; LatentMoE hybrid (Mamba-2 + MoE + Attention) |

FP8 chosen over MXFP4 for the Qwen path (avoids marlin/CUTLASS/FlashInfer-sinks issues on GB10/SM121). The Nemotron path uses NVFP4 which is a natural fit for the GB10 FP4 tensor cores — do not switch its `--quantization` flag.

Nemotron usage. The Ray cluster must be **brought up with Nemotron-specific env vars present in each container's start-time env**; Ray cannot propagate vLLM runtime flags from the head driver to worker ranks at vllm-serve time, so the workers would otherwise pick mismatched FP4 kernels / allreduce backends. Source `nemotron/cluster-env.sh` on **both** nodes before launching the bring-up scripts:

```bash
# Node 1 (head)
source nemotron/cluster-env.sh
cd cluster/head && bash run_headnode_2.sh

# Node 2 (worker)
source nemotron/cluster-env.sh
cd cluster/worker && bash run_workernode_2.sh

# Node 1 again, new terminal
cd nemotron
# .env needs HF_TOKEN and VLLM_API_KEY
./launch-nemotron-120b.sh

# Override knobs via env (see top of script):
MAX_MODEL_LEN=1048576 GPU_MEM_UTIL=0.80 ENABLE_MTP=1 ./launch-nemotron-120b.sh   # push to 1M after 512k proves stable
ENABLE_EAGER=1 ./launch-nemotron-120b.sh   # skip CUDA graph capture if memory spikes on first inference
```

Equivalent via the top-level Makefile (sources `nemotron/cluster-env.sh` automatically inside `make head` / `make worker`):

```bash
make head       # Node 1
make worker     # Node 2
make nemotron   # Node 1, new terminal
```

The launcher refuses to run if the four `VLLM_NVFP4_*` / `VLLM_FLASHINFER_*` / `VLLM_USE_FLASHINFER_MOE_FP4` / `VLLM_ALLOW_LONG_MAX_MODEL_LEN` vars are missing from the head container's env — it prints the tear-down/bring-up steps and exits non-zero.

The Qwen path has no equivalent `cluster-env.sh` because its FP8 deployment requires no `VLLM_*` runtime-env overrides at container-start time — vLLM picks correct kernels and allreduce backends from its FP8 defaults. The cluster bring-up scripts treat `VLLM_FORWARD_VARS` as empty when unset, so you can bring the same cluster up for Qwen without sourcing anything.

Nemotron host-stability safeguards:
- `--gpu-memory-utilization 0.75` (conservative; NVIDIA's example uses 0.9).
- `--max-model-len 1048576` (1M tokens, model maximum). Verified stable on this hardware after a 512k checkpoint (host `MemAvailable` ~18 GB during inference at 512k). Drop to `MAX_MODEL_LEN=524288` or `262144` via env if you hit memory pressure under heavier concurrent load.
- TP=2 across both Sparks roughly halves per-node weight memory vs. single-Spark.

The Ray container is launched by `cluster/head/run_cluster.sh` **without** a `--memory` cgroup cap, so this launcher cannot add one. Host-side hardening on **both** Sparks is therefore not optional — the earlier `gpt-oss-120b` single-Spark crash that bricked sshd while ICMP still replied is the reason for the defenses listed below.

Host-side hardening recommended once, outside this repo:
- `sudo systemctl edit ssh` → add `[Service]\nOOMScoreAdjust=-1000`
- `sudo apt install earlyoom && sudo systemctl enable --now earlyoom`
- External watchdog (laptop): `curl :8000/health` every 30 s; IPMI/PDU-cycle on N failures.

## Networking knobs (passed into containers)

| Env var | Value | Why |
|---|---|---|
| `PRIMARY_IF` | `enp1s0f0np0` | Ray control + TP rendezvous (single IP); the `f0` half that stays up after cold *and* warm boots |
| `DATA_IFS` | carrier-up ports, comma-separated | UCX, NCCL sockets, Gloo, OMPI TCP; built at bring-up by `select_up_dataplane` (4 cold / 2 warm) |
| `NCCL_IB_HCA` | carrier-up RoCE HCAs (`rocep1s0f0,roceP2p1s0f0` + the `f1` HCAs when up) | RoCE HCAs for NCCL RDMA |
| `UCX_NET_DEVICES` | same HCAs with `:1` port suffix | UCX RDMA transport (not netdev names) |
| `NCCL_CROSS_NIC` | `1` | Allow cross-NIC pairing |
| `TP_SOCKET_IFNAME` | `$PRIMARY_IF` | PyTorch TP needs a single IP, not the data plane list |
| `RAY_memory_monitor_refresh_ms` | `0` | Disable Ray OOM killer (vLLM manages its own memory) |

Optional verbose NCCL logging (off by default):
```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET bash run_headnode_2.sh
```

## Shutdown

`Ctrl-C` on the head terminal stops the head container. **The worker container does not exit on its own** — Ray loses its head but `ray start --block` keeps running. On the worker:

```bash
docker stop $(docker ps --format '{{.Names}}' | grep '^node-')
```

## Host maintenance: kernel upgrades and reboots

The Sparks have **no DKMS**. NVIDIA kernel modules come only from prebuilt `linux-modules-nvidia-<branch>-<kernel>` packages, and the kernel (src: `linux-nvidia`) and the driver (src: `nvidia-graphics-drivers-<branch>`) are separate source packages on independent *phased-update* schedules — with phasing decided **per-machine**. A single `apt upgrade` can therefore pull a new kernel while holding the NVIDIA driver back, on one Spark but not the other. Reboot into that gap and the node comes up with no `nvidia.ko` at all: `nvidia-smi` dead, no `/dev/nvidia*`, and every GPU container failing in the prestart hook.

Run on **each** node before rebooting it:

```bash
make preboot-check     # SAFE / UNSAFE TO REBOOT
```

`UNSAFE` means the kernel GRUB will boot **next** has no NVIDIA modules. Resync before rebooting:

```bash
make fix-nvidia        # installs the matching modules metapackage, then loads it
```

Recovery needs no reboot — the modules target the already-running kernel. Expect `fix-nvidia` to move the whole NVIDIA userspace to a new driver version; that is correct rather than collateral damage, since the modules package hard-depends on a matching `nvidia-kernel-common-<branch>`, so module and userspace advance together by construction.

Health check at any time, per node:

```bash
make check-nvidia
```

Exit codes: `0` healthy · `1` driver unusable now (blocks `make head` / `make worker`) · `3` driver fine but the metapackages have drifted, so the **next** reboot is the risk (warns, does not block) · `2` the check itself could not run.

**Both Sparks must report the same driver version** before Ray comes up — don't leave a split-version pair across the RoCE fabric.

## Troubleshooting

- **Slow inter-node AllReduce / NCCL falls back to TCP.** Check that `/dev/infiniband/uverbs*` exists inside the worker container (`docker exec <node> ibv_devinfo`). Both `run_cluster.sh` copies must pass `--device=/dev/infiniband --cap-add=IPC_LOCK --ulimit memlock=-1:-1`; they are kept byte-identical for this reason.
- **`nvidia-container-cli: initialization error: nvml error: driver not loaded`** on `make head` / `make worker`. The host NVIDIA driver is not loaded, usually because a kernel upgrade landed a kernel with no matching modules package — see [Host maintenance](#host-maintenance-kernel-upgrades-and-reboots). Diagnose with `make check-nvidia`, repair with `make fix-nvidia`. Distinct from the cgroup-revocation failure, which shows `Failed to initialize NVML: Unknown Error` while the host driver is loaded fine.
- **`No node-* container running`** from launch script. Head container not up yet, or `docker ps` filter misses it (custom name). Start head first.
- **122B OOM on first inference.** Drop `--gpu-memory-utilization` from 0.85 to 0.80 in `launch-qwen-122b.sh`, or reduce `--max-model-len`.
- **`WARNING: Using default MoE config ... GB10.json`.** Harmless. No hand-tuned MoE kernel config for GB10 yet; auto defaults work.
- **Host freezes during inference.** A locally-run memory sampler dies with the host. Run a laptop-side `free`/`vmstat` poller over SSH for freeze detection.

## See also

- `CLAUDE.md` — operator notes for Claude Code working on this repo.
