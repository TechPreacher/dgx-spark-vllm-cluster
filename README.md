# dgx-spark-vllm-cluster

Operator scripts for a 2-node NVIDIA DGX Spark cluster running distributed LLM inference via Ray + vLLM in Docker.

## Hardware

- 2× NVIDIA DGX Spark (GB10 / SM121, 128 GB unified memory each)
- 2× 200 GbE ConnectX-7 NICs per node → 4 ports total, ~800 GbE aggregate data plane (RoCE)
- One control-plane interface (`enp1s0f1np1`) on a `/24` carrying Ray + TP rendezvous
- Head node primary IP: `10.0.1.3` (override via `HEAD_NODE_IP` env)

## Prerequisites

- Docker with NVIDIA runtime
- HuggingFace cache at `~/.cache/huggingface` (writable, shared via bind mount)
- `qwen/.env` with `VLLM_API_KEY=…` (not committed — see `.gitignore`)
- Optional pre-stage: `hf download Qwen/Qwen3-30B-A3B-Thinking-2507-FP8`

## Layout

```
cluster/
  head/
    run_headnode_2.sh        # 4-port data plane, current default
    run_headnode.sh          # single-interface fallback
    run_cluster.sh           # generic Ray+Docker launcher (head/worker)
    ray_inference_health.sh  # ray status + /health + nvidia-smi
  worker/
    run_workernode_2.sh      # 4-port data plane, current default
    run_workernode.sh        # single-interface fallback
    run_cluster.sh           # byte-identical to head/run_cluster.sh
qwen/
  launch-qwen-30b.sh         # Qwen3-30B-A3B-Thinking-2507-FP8 (TP=2, 131k ctx)
  launch-qwen-122b.sh        # Qwen3.5-122B-A10B-FP8 (TP=2, 65k ctx, Qwen3-Next hybrid)
CLAUDE.md                    # operator notes for Claude Code
```

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

FP8 chosen over MXFP4 to avoid marlin/CUTLASS/FlashInfer-sinks issues on GB10 (SM121). Do not pass `--quantization` — auto-detected.

## Networking knobs (passed into containers)

| Env var | Value | Why |
|---|---|---|
| `PRIMARY_IF` | `enp1s0f1np1` | Ray control + TP rendezvous (single IP) |
| `DATA_IFS` | 4 ports comma-separated | UCX, NCCL sockets, Gloo, OMPI TCP |
| `NCCL_IB_HCA` | `rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1` | RoCE HCAs for NCCL RDMA |
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

## Troubleshooting

- **Slow inter-node AllReduce / NCCL falls back to TCP.** Check that `/dev/infiniband/uverbs*` exists inside the worker container (`docker exec <node> ibv_devinfo`). Both `run_cluster.sh` copies must pass `--device=/dev/infiniband --cap-add=IPC_LOCK --ulimit memlock=-1:-1`; they are kept byte-identical for this reason.
- **`No node-* container running`** from launch script. Head container not up yet, or `docker ps` filter misses it (custom name). Start head first.
- **122B OOM on first inference.** Drop `--gpu-memory-utilization` from 0.85 to 0.80 in `launch-qwen-122b.sh`, or reduce `--max-model-len`.
- **`WARNING: Using default MoE config ... GB10.json`.** Harmless. No hand-tuned MoE kernel config for GB10 yet; auto defaults work.
- **Host freezes during inference.** A locally-run memory sampler dies with the host. Run a laptop-side `free`/`vmstat` poller over SSH for freeze detection.

## See also

- `CLAUDE.md` — operator notes for Claude Code working on this repo.
