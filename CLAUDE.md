# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Bash scripts (no application code) that operate a 2-node NVIDIA DGX Spark (GB10 / SM121) Ray cluster running vLLM inside Docker for distributed LLM inference. The two Sparks are linked by 2× 200 GbE ConnectX-7 NICs (4 ports total, ~800 GbE aggregate data plane). Models tested: Qwen3-30B-A3B-Thinking-2507-FP8 and Qwen3.5-122B-A10B-FP8 across both nodes; Qwen3.6-35B-A3B-FP8 single-Spark via Docker Compose.

## Bring-up sequence (must run in order)

1. **Head node (Node 1, `10.0.1.3`):** `cd cluster/head && bash run_headnode_2.sh`
2. **Worker node (Node 2):** `cd cluster/worker && bash run_workernode_2.sh`
3. **Inject model into running head container:** `cd qwen && ./launch-qwen-30b.sh` (or `./launch-qwen-122b.sh`)

Each `run_*node_2.sh` script blocks (it `docker run`s with the Ray `--block` command). Closing the terminal stops Ray on that node and tears down the whole cluster. The model-launch script `docker exec`s `vllm serve` into the head container; Ray then schedules TP shard 2 onto Node 2 automatically (`--tensor-parallel-size 2`).

`launch-qwen-*.sh` requires `qwen/.env` containing `VLLM_API_KEY=...` (gitignored — do not commit). They find the head container by matching `^node-[0-9]+$` from `docker ps`.

## The two cluster variants

- **`run_headnode_2.sh` / `run_workernode_2.sh` (current, in use):** 4-port data plane. `PRIMARY_IF=enp1s0f1np1` carries Ray control (single IP for `VLLM_HOST_IP` / `MASTER_ADDR`). `DATA_IFS=enp1s0f0np0,enp1s0f1np1,enP2p1s0f0np0,enP2p1s0f1np1` is exported to `UCX_NET_DEVICES`, `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, and `OMPI_MCA_btl_tcp_if_include`. NCCL is told the 4 RoCE HCAs via `NCCL_IB_HCA=rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1` with `NCCL_CROSS_NIC=1`. `TP_SOCKET_IFNAME` is pinned to `PRIMARY_IF` so PyTorch TP setup uses the control IP.
- **`run_headnode.sh` / `run_workernode.sh` (older, single-interface):** Only `enp1s0f1np1`, no IB env. Kept for fallback; do not edit when the 4-port path is in use.

Both call `cluster/{head,worker}/run_cluster.sh`. The head-side copy adds `--device=/dev/infiniband --cap-add=IPC_LOCK --ulimit memlock=-1:-1` to the `docker run` — the worker-side copy currently does not. Keep that in mind if you change one: edit both unless you know why they differ.

`run_cluster.sh` positional args: `<image> <head_node_ip> --head|--worker <hf_cache_path> [extra docker args...]`. It extracts `VLLM_HOST_IP` from the extra args, names the container `node-${RANDOM}`, and traps `EXIT` to `docker stop && docker rm` on script exit.

Image: `nvcr.io/nvidia/vllm:25.11-py3` for the Ray cluster path; `nvcr.io/nvidia/vllm:26.05.post1-py3` for the single-Spark Compose path. HuggingFace cache is bind-mounted from `~/.cache/huggingface`.

## Model launch — what to know before editing

The Qwen launch scripts encode tuning that is non-obvious:

- **30B Thinking (FP8):** TP=2, ctx 131072, `gpu-memory-utilization 0.70`, `--reasoning-parser deepseek_r1`, `--tool-call-parser hermes`.
- **122B A10B (FP8, Qwen3-Next hybrid Gated DeltaNet + Gated Attention MoE):** TP=2, ctx cut to **65536** (vs native 262k) and util raised to **0.85** because FP8 weights ~125 GB → ~63 GB/node on 128 GB unified memory leaves little headroom for KV+CUDA graphs. Parsers change to `--reasoning-parser qwen3` and `--tool-call-parser qwen3_coder`. `--trust-remote-code` required.
- **35B A3B (FP8) single-Spark Compose:** TP=1, ctx 32768, `kv-cache-dtype fp8`. FP8 (not MXFP4) is chosen to avoid marlin/CUTLASS/FlashInfer-sinks SM121 (GB10) issues; the comment in `qwen/compose.yml` warns that the `"Config file not found ... GB10.json"` MoE warning is harmless. Do not pass `--quantization` — auto-detected from FP8 repos.

## Health / monitoring

- `cluster/head/ray_inference_health.sh` — `ray status` in container, `curl :8000/health`, `nvidia-smi` on host + in container.
- `cluster/{head,worker}/mem-watch.sh` — local `free -m` sampler. Comment warns: runs on the Spark itself, so a host freeze takes the watcher with it; use a laptop-side equivalent for freeze detection.

## Things that look risky and aren't (and vice-versa)

- The `Warning: VLLM_HOST_IP differs from head_node_ip` branch in `run_cluster.sh` resolves by trusting `VLLM_HOST_IP` — intentional.
- `RAY_memory_monitor_refresh_ms=0` disables Ray's OOM killer; deliberate for vLLM workloads.
- `TP_SOCKET_IFNAME=$PRIMARY_IF` (not `$DATA_IFS`) is intentional — PyTorch TP rendezvous needs a single IP; data plane is for NCCL/UCX.
- `update.sh` files only update host packages (brew, pipx). They have nothing to do with the cluster.
