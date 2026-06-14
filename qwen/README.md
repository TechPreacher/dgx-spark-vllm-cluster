# Qwen on the 2-Spark Ray cluster

Two launchers, both inject the model into the running Ray container via `docker exec`. Pick one; do not run both at the same time (port 8000 collision).

| Script | Model | TP | ctx |
|---|---|---|---|
| `launch-qwen-30b.sh`  | `Qwen/Qwen3-30B-A3B-Thinking-2507-FP8` | 2 | 131072 |
| `launch-qwen-122b.sh` | `Qwen/Qwen3.5-122B-A10B-FP8`           | 2 | 65536  |

## One-time setup

```bash
cp .env.example .env
$EDITOR .env                    # set VLLM_API_KEY (HF_TOKEN only needed for first hf download)
# Optional: pre-stage weights to avoid slow first launch.
hf download Qwen/Qwen3-30B-A3B-Thinking-2507-FP8
```

## Launch (in this order)

```bash
# Node 1 (head)
cd ../cluster/head && bash run_headnode_2.sh

# Node 2 (worker)
cd ../cluster/worker && bash run_workernode_2.sh

# Node 1, new terminal
cd ../qwen
./launch-qwen-30b.sh            # or ./launch-qwen-122b.sh
```

Do **not** source `nemotron/cluster-env.sh` for the Qwen path — the FP8 deployment needs no `VLLM_*` runtime overrides.

## Smoke test

```bash
curl -fsS http://<head-ip>:8000/health && echo
curl http://<head-ip>:8000/v1/chat/completions \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3_30b_thinking","messages":[{"role":"user","content":"12*17"}],"max_tokens":200}'
```

Served model names: `qwen3_30b_thinking` (30B) / `qwen35_122b_thinking` (122B).

## Shutdown

`Ctrl-C` the launcher (stops `vllm serve`; Ray container keeps running, so you can switch models without tearing the cluster down). To stop the cluster too: `Ctrl-C` the head terminal, then on the worker `docker stop $(docker ps --format '{{.Names}}' | grep '^node-')`.
