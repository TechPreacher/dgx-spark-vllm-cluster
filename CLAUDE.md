# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Bash scripts (no application code) that operate a 2-node NVIDIA DGX Spark (GB10 / SM121) Ray cluster running vLLM inside Docker for distributed LLM inference. The two Sparks are linked by 2× 200 GbE ConnectX-7 NICs (4 ports total, ~800 GbE aggregate data plane). Models tested: Qwen3-30B-A3B-Thinking-2507-FP8 and Qwen3.5-122B-A10B-FP8 across both nodes. A third launcher (`nemotron/launch-nemotron-120b.sh`) runs NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 across the same 2-node Ray cluster (TP=2) using native NVFP4 on SM121 FP4 tensor cores.

## Bring-up sequence (must run in order)

1. **Head node (Node 1, `10.0.1.3`):** `cd cluster/head && bash run_headnode_2.sh`
2. **Worker node (Node 2):** `cd cluster/worker && bash run_workernode_2.sh`
3. **Inject model into running head container:** `cd qwen && ./launch-qwen-30b.sh` (or `./launch-qwen-122b.sh`)

Each `run_*node_2.sh` script blocks (it `docker run`s with the Ray `--block` command). Closing the terminal stops Ray on that node and tears down the whole cluster. The model-launch script `docker exec`s `vllm serve` into the head container; Ray then schedules TP shard 2 onto Node 2 automatically (`--tensor-parallel-size 2`).

`launch-qwen-*.sh` requires `qwen/.env` containing `VLLM_API_KEY=...` (gitignored — do not commit). They find the head container by matching `^node-[0-9]+$` from `docker ps`.

## The two cluster variants

- **`run_headnode_2.sh` / `run_workernode_2.sh` (current, in use):** 4-port data plane. `PRIMARY_IF=enp1s0f1np1` carries Ray control (single IP for `VLLM_HOST_IP` / `MASTER_ADDR`). `DATA_IFS=enp1s0f0np0,enp1s0f1np1,enP2p1s0f0np0,enP2p1s0f1np1` is exported to `UCX_NET_DEVICES`, `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, and `OMPI_MCA_btl_tcp_if_include`. NCCL is told the 4 RoCE HCAs via `NCCL_IB_HCA=rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1` with `NCCL_CROSS_NIC=1`. `TP_SOCKET_IFNAME` is pinned to `PRIMARY_IF` so PyTorch TP setup uses the control IP.
- **`run_headnode.sh` / `run_workernode.sh` (older, single-interface):** Only `enp1s0f1np1`, no IB env. Kept for fallback; do not edit when the 4-port path is in use.

Both call `cluster/{head,worker}/run_cluster.sh`. The two copies are kept byte-identical (`--device=/dev/infiniband --cap-add=IPC_LOCK --ulimit memlock=-1:-1` on both, `--shm-size 16g`). If you edit one, copy to the other — RoCE/IB device passthrough must exist on both sides or NCCL silently falls back to TCP and you lose the ~800 GbE data plane.

`run_cluster.sh` positional args: `<image> <head_node_ip> --head|--worker <hf_cache_path> [extra docker args...]`. It extracts `VLLM_HOST_IP` from the extra args, names the container `node-${RANDOM}`, and traps `EXIT` to `docker stop && docker rm` on script exit. Caveat: Ctrl-C on the head terminal stops only the head container; the worker `ray start --block` keeps running attached to a now-dead head — `docker stop node-*` on the worker by hand to clean up.

Optional verbose NCCL logging (off by default): prefix the launch with `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET bash run_*node_2.sh`. Worker head IP override: `HEAD_NODE_IP=10.0.1.x bash run_workernode_2.sh`.

Image: `local/vllm-ray:26.05.post1` — a locally-built tag layered on top of `nvcr.io/nvidia/vllm:26.05.post1-py3` (NGC dropped Ray from the 26.05 vLLM image; `cluster/Dockerfile` adds it back). Build with `bash cluster/build-image.sh` on each node before first use; rebuild after bumping `BASE_IMAGE`. Override per-bring-up with `VLLM_IMAGE=<tag>`. HuggingFace cache is bind-mounted from `~/.cache/huggingface`. Single image for the whole cluster — Qwen and Nemotron share it.

## Model launch — what to know before editing

The Qwen launch scripts encode tuning that is non-obvious:

- **30B Thinking (FP8):** TP=2, ctx 131072, `gpu-memory-utilization 0.70`, `--reasoning-parser deepseek_r1`, `--tool-call-parser hermes`.
- **122B A10B (FP8, Qwen3-Next hybrid Gated DeltaNet + Gated Attention MoE):** TP=2, ctx cut to **65536** (vs native 262k) and util raised to **0.85** because FP8 weights ~125 GB → ~63 GB/node on 128 GB unified memory leaves little headroom for KV+CUDA graphs. Parsers change to `--reasoning-parser qwen3` and `--tool-call-parser qwen3_coder`. `--trust-remote-code` required.
FP8 (not MXFP4) is chosen for the Qwen path to avoid marlin/CUTLASS/FlashInfer-sinks SM121 (GB10) issues. Do not pass `--quantization` on Qwen — vLLM auto-detects from FP8 repos. The `"Config file not found ... GB10.json"` MoE warning at load is harmless (no hand-tuned MoE kernel config for GB10 yet).

## Nemotron-3-Super-120B-A12B-NVFP4 (Ray TP=2 across both Sparks)

`nemotron/launch-nemotron-120b.sh` follows the same pattern as the Qwen launchers: `docker exec` into the running `^node-[0-9]+$` head container and `vllm serve` with `--tensor-parallel-size 2`, letting Ray dispatch shard 2 to the worker over the 800 GbE data plane. It does NOT do a `docker run` of its own — the Ray cluster must already be up (`run_headnode_2.sh` + `run_workernode_2.sh`).

The model is NVFP4-quantized (native GB10/SM121 FP4 tensor cores) and is a LatentMoE hybrid (Mamba-2 + MoE + Attention). `--mamba-ssm-cache-dtype float16` matters; the reasoning parser is a plugin file (`super_v3_reasoning_parser.py`) the launcher fetches **inside the container** at exec time and stores under `~/.cache/huggingface/` (which is bind-mounted, so it survives Ray container restarts). No host-side bind mount of the parser is needed.

Required env (set in `nemotron/.env`): `HF_TOKEN`, `VLLM_API_KEY`. Unlike the Qwen launchers, `--quantization fp4` and `--moe-backend marlin` are passed explicitly per NVIDIA's DGX Spark example.

**Critical bring-up requirement** that the Qwen path does not have: the four NVFP4 runtime env vars (`VLLM_NVFP4_GEMM_BACKEND=marlin`, `VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm`, `VLLM_USE_FLASHINFER_MOE_FP4=0`, `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`) must be present in each Ray container's env at `docker run` time, **on both nodes**. Ray does not propagate `os.environ` from the head driver to worker ranks across nodes; rank 1 in the worker container therefore inherits only that container's start-time env. If the four vars are missing on the worker, rank 1 picks a different FP4 GEMM kernel / allreduce backend than rank 0 → mismatch, collective hang, or crash at first matmul. The launcher refuses to run if it doesn't see them inside the head container and prints the bring-up instructions.

The mechanism: both `run_headnode_2.sh` and `run_workernode_2.sh` now read a space-separated `VLLM_FORWARD_VARS` from the parent shell and forward each named var into the container as `-e VAR=value`. `nemotron/cluster-env.sh` exports the four NVFP4 vars and sets `VLLM_FORWARD_VARS` to enumerate them. Both nodes must `source nemotron/cluster-env.sh` before running the bring-up script. To add new model-specific runtime flags in the future, append their names to `VLLM_FORWARD_VARS` rather than editing the bring-up scripts.

There is intentionally no `qwen/cluster-env.sh` — the Qwen FP8 path requires no `VLLM_*` runtime-env overrides at container-start time. Bringing the cluster up without sourcing any profile is the correct flow for Qwen. The passthrough loop is a no-op when `VLLM_FORWARD_VARS` is unset.

**Host-stability context:** an earlier `gpt-oss-120b` single-Spark run starved the host of memory so badly that sshd became unreachable while ICMP still replied; recovery required a power cycle. With TP=2, per-node weight memory is roughly halved versus that single-Spark run, but the Ray container has no `--memory` cgroup cap (that's set by `run_cluster.sh` at container create time, not editable after the fact). Defences here are: `--gpu-memory-utilization 0.75` (vs NVIDIA's 0.9), `--max-model-len 262144` (vs 1M), and an opt-in `ENABLE_EAGER=1` that skips CUDA graph capture if first-inference memory spikes are a concern. MTP speculative decoding is off by default; enable with `ENABLE_MTP=1`.

Out-of-repo hardening that complements the launcher (apply on **both** Sparks): `OOMScoreAdjust=-1000` on sshd via `systemctl edit ssh`, `earlyoom` package installed and enabled, plus an external laptop-side watchdog that IPMI/PDU-cycles on repeated `/health` failures.

## Health / monitoring

- `cluster/head/ray_inference_health.sh` — `ray status` in container, `curl :8000/health`, `nvidia-smi` on host + in container. Exits non-zero if no `node-*` container is running.

## Things that look risky and aren't (and vice-versa)

- The `Warning: VLLM_HOST_IP differs from head_node_ip` branch in `run_cluster.sh` resolves by trusting `VLLM_HOST_IP` — intentional.
- `RAY_memory_monitor_refresh_ms=0` disables Ray's OOM killer; deliberate for vLLM workloads.
- `TP_SOCKET_IFNAME=$PRIMARY_IF` (not `$DATA_IFS`) is intentional — PyTorch TP rendezvous needs a single IP; data plane is for NCCL/UCX.
- `UCX_NET_DEVICES` uses RDMA device names with `:1` port suffix (e.g. `rocep1s0f0:1`), not netdev names — UCX needs the RDMA device path to use the RoCE transport rather than falling back to TCP.
- `CONTAINER_NAME="node-$(date +%s)$$"` keeps the `^node-[0-9]+$` shape that all the launcher / health scripts grep for, while avoiding the `$RANDOM` (15-bit) collision risk across rapid reruns.
