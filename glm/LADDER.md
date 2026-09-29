# GLM-5.3-Flash context ladder — measurement log

Status: **rungs 1 and 3 PASS on both nodes 2026-09-29 — the 262K target is
served and verified.** Rung 4 (DFlash2) and rung 5 (CUTLASS MoE) are outstanding.
Prerequisites were verified green 2026-09-28. Fill each rung in as it is climbed;
do not skip rungs.

Headroom here is ~12.9 GiB/node against Nemotron's roughly double, and this
cluster has a documented memory-starvation failure (`gpt-oss-120b`) that took
sshd unreachable while ICMP still replied and needed a power cycle to recover.
That is why the ladder exists and why the prerequisites are prerequisites.

## Prerequisites — run on BOTH nodes before rung 1

Verified on both nodes 2026-09-28:

| Check | pulsar | magnetar | Required |
|---|---|---|---|
| `earlyoom` | active + enabled | active + enabled | active + enabled |
| earlyoom SIGTERM | `mem<=4%`, swap-independent | `mem<=4%`, swap-independent | swap-independent |
| earlyoom SIGKILL | `mem<=2%`, swap-independent | `mem<=2%`, swap-independent | swap-independent |
| sshd listener `oom_score_adj` | `-1000` | `-1000` | `-1000` |
| `vm.swappiness` | `0` (persisted) | `0` (persisted) | `0` |
| docker cgroup driver | `cgroupfs` | `cgroupfs` | `cgroupfs` |
| NVIDIA driver | 580.178.04 | 580.178.04 | identical both nodes |
| `check_nvidia.sh` | exit 0 | exit 0 | exit 0 |
| GLM image verified | pass | pass | pass |

```bash
# On EACH node:
sudo apt install earlyoom && sudo systemctl enable --now earlyoom
sudo systemctl edit ssh          # add:  [Service]\nOOMScoreAdjust=-1000
sudo systemctl restart ssh
echo 'vm.swappiness=0' | sudo tee /etc/sysctl.d/99-glm-uvm.conf
sudo sysctl -w vm.swappiness=0

# Immediately before each rung, on EACH node:
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
free -g
```

`vm.swappiness=0` is the UVM-livelock defence on GB10 unified memory. Note one
EXL3-based recipe recommends `180` plus zram instead; that is a different stack
(EXL3/TR3, not NVFP4 weight-only) and is deliberately not adopted here.

### earlyoom must be reconfigured, or it is inert here

**earlyoom's stock thresholds are an AND across memory and swap:**

```
SIGTERM when mem <= 10.00% and swap <= 10.00%,
SIGKILL when mem <=  5.00% and swap <=  5.00%
```

With `vm.swappiness=0` the 16 GB of swap stays essentially unused, so the swap
condition is never met and **earlyoom never fires** under GPU-driven memory
pressure. The two prerequisites fight each other: `swappiness=0` is precisely
what disarms the stock earlyoom. Installing it and stopping there gives false
comfort.

Required on both nodes:

```bash
sudo sed -i 's/^EARLYOOM_ARGS=.*/EARLYOOM_ARGS="-r 3600 -m 4,2 -s 100,100"/' /etc/default/earlyoom
sudo systemctl restart earlyoom
sleep 2   # journald lag: reading immediately after restart returns the PREVIOUS run's lines
journalctl -u earlyoom --no-pager --since '-2min' | grep -E 'SIGTERM|SIGKILL' | tail -2
```

The `--since` window plus the short wait matter: `journalctl -n N` straight after
a restart can return the *previous* startup's thresholds, which reads exactly like
the change having failed. Confirm by the timestamp, not just the values.

`-s 100,100` makes the swap side always true for **both** signals, reducing each
AND to its memory condition. `-m 4,2` puts SIGTERM at 4% of 124608 MiB ≈ **4.9
GiB** and SIGKILL at 2% ≈ **2.5 GiB**.

Both kill percentages must be given explicitly. With a bare `-m 4 -s 100`,
earlyoom halves *both* percentages for SIGKILL and you get `mem <= 2.00% and
swap <= 50.00%` — and since `swappiness=0` keeps swap ~100% free, that SIGKILL
can never fire. SIGTERM would still work, but the escalation path would be
disarmed for precisely the case it exists for: a process wedged in UVM livelock
that does not respond to SIGTERM.

4.9 GiB sits just above this document's 4 GB abort criterion, so earlyoom becomes
the **automatic backstop for exactly that threshold**. The stock 10% (≈12.2 GiB)
is too aggressive: it is high enough to kill a healthy run, and the Nemotron notes
record normal operation at ~18 GB available.

**Consequence for reading rung failures:** once configured this way, a rung that
exhausts memory presents as **vLLM being SIGTERMed**, not as a hang or an
unreachable host. Check `journalctl -u earlyoom` and `MemAvailable` before
suspecting the model or the image.

Also required: `glm/.env` with `HF_TOKEN` and `VLLM_API_KEY`.

Neither model repo is gated (`gated: false` on both `LibertAIDAI/GLM-5.3-Flash-NVFP4`
and `incoai/GLM-5.3-Flash-DFlash2`), so **there are no terms to accept** — any
read-scoped token works, and it is only there for pull rate limits and the
launcher's `:?` guard. Reuse the one in `nemotron/.env` if you like.

The drafter's `cc-by-nc-nd-4.0` still binds **use** regardless: ungated means
nothing blocks the download, not that the terms lapse. Non-commercial, no
redistribution.

First run downloads ~181 GiB into `~/.cache/huggingface` (381 GB already used,
3.1 TB free — fits).

## Rungs

| Rung | ctx | util | KV | Spec | Proves | MemAvail pulsar | MemAvail magnetar | decode tok/s | Result |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 32K | n/a | fp8, 6 GiB | off | Weights load; TP2 collectives alive across RoCE | 8.1 GB | 12.4 GB | ~14.7 | **PASS** |
| 2 | 131K | n/a | fp8, 6 GiB | off | KV math holds | — | — | — | _skipped — went straight to 262K_ |
| 3 | 262K | n/a | fp8, 6 GiB | off | Target context | 6.6 GB idle / 6.1 GB under 60K load | 11.1 GB | ~14.7 | **PASS** |
| 4 | 262K | n/a | fp8, **3 GiB** | dflash, 7 | Acceptance + tok/s vs published 46.9 / 74.1% | | | | _ready to run_ |
| 5 | 262K | n/a | fp8, 6 GiB | off, **CUTLASS MoE** | FP4 tensor cores vs marlin's weight-only 14.7 | | | | _building_ |

### Abort criteria — any one, on either node

- host `MemAvailable` below ~4 GB
- sshd latency degrading
- `dmesg -T | tail -30` showing UVM or OOM activity

If a rung trips these, the **previous** rung is the working configuration.
Record it and stop; do not press on.

### Commands

```bash
# Rung 1
source glm/cluster-env.sh && make head PROFILE=glm       # Node 1
source glm/cluster-env.sh && make worker PROFILE=glm     # Node 2
MAX_MODEL_LEN=32768  GPU_MEM_UTIL=0.80 make serve PROFILE=glm   # Node 1, new terminal

# Rung 2 / 3  (tear down both ranks first: docker stop node-* on BOTH nodes)
MAX_MODEL_LEN=131072 GPU_MEM_UTIL=0.85 make serve PROFILE=glm
MAX_MODEL_LEN=262144 GPU_MEM_UTIL=0.85 make serve PROFILE=glm

# Rung 4
MAX_MODEL_LEN=262144 GPU_MEM_UTIL=0.85 ENABLE_DFLASH2=1 make serve PROFILE=glm
```

Watch during load, on both nodes: `watch -n5 'grep MemAvailable /proc/meminfo'`

### Smoke test (every rung)

```bash
source glm/.env
curl -s http://localhost:8000/health && echo " health OK"
curl -s http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer ${VLLM_API_KEY}" -H 'Content-Type: application/json' \
  -d '{"model":"zai-org/glm-5.3-flash","messages":[{"role":"user","content":"Reply with exactly: ok"}],"max_tokens":16}'
```

### Long-context check (rung 3)

Proves a long prompt survives the fp8 KV budget rather than silently truncating.

```bash
source glm/.env
python3 - <<'PY'
import json, subprocess, os
prompt = "The magic word is 'zarquon'. " + ("filler text. " * 20000) + " What is the magic word?"
body = json.dumps({"model":"zai-org/glm-5.3-flash",
                   "messages":[{"role":"user","content":prompt}],"max_tokens":32})
out = subprocess.run(["curl","-s","http://localhost:8000/v1/chat/completions",
  "-H",f"Authorization: Bearer {os.environ['VLLM_API_KEY']}",
  "-H","Content-Type: application/json","-d",body],capture_output=True,text=True).stdout
print(out[:600])
PY
```

Expected: the response recovers `zarquon`.

### Decode baseline (rung 3, then compare at rung 4)

```bash
source glm/.env
time curl -s http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer ${VLLM_API_KEY}" -H 'Content-Type: application/json' \
  -d '{"model":"zai-org/glm-5.3-flash","messages":[{"role":"user","content":"Write a Python function that merges two sorted lists. Explain it."}],"max_tokens":400}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["usage"])'
```

## Rung 1 — PASS (2026-09-29)

Reached KV-cache allocation successfully:

```
GPU KV cache size: 449,114 tokens
Maximum concurrency for 32,768 tokens per request: 13.71x
Initial free memory: 106.94 GiB (pulsar), 110.97 GiB (magnetar)
reserved 6.0 GiB for KV Cache as specified by kv_cache_memory_bytes
```

First real inference confirmed: `17 × 23 = 391`, correct and cleanly formatted,
324 completion tokens in 22.0 s ≈ **14.7 tok/s** decode (eager, marlin
weight-only, no speculation). Steady-state host `MemAvailable` with the model
resident: **pulsar 8.1 GB, magnetar 12.4 GB**.

That pulsar figure is the number to watch. earlyoom SIGTERMs at ~4.9 GB, so the
margin at 32K is only ~3.2 GB, and pulsar consistently runs ~4 GB tighter than
magnetar. Rungs 2 and 3 should be read against this, not against the 113 GB
idle baseline.

**449,114 tokens at 6 GiB fp8** is the headline KV number, and it is very good news
for rung 3: at 262,144 tokens per request that is still **1.7x concurrency**, so
the 262K target looks reachable on KV grounds. It also means 6 GiB is generous at
32K — dropping `KV_CACHE_MEMORY` is a spare lever if host memory stays tight.

**`GPU_MEM_UTIL` is inert in this configuration.** vLLM logs it explicitly:
"reserved 6.0 GiB ... as specified by kv_cache_memory_bytes config and skipped
memory profiling. This does not respect the gpu_memory_utilization config."
Setting `kv_cache_memory_bytes` bypasses profiling entirely, so the util knob
does nothing unless that is unset. The ladder's util column is therefore
descriptive, not causal, while `KV_CACHE_MEMORY` is set.

Note also the marlin backend logs "Your GPU does not have native support for FP4
computation ... Weight-only FP4 compression will be used leveraging the Marlin
kernel." That is expected and correct here: the LibertAI checkpoint is
**weight-only NVFP4-A16**, so weight-only decompression is the right path, not a
fallback. It does mean marlin is not exercising the FP4 tensor cores, which is
what `flashinfer_cutlass` would do once the nvrtc.h image fix is rebuilt.

## Rung 3 — PASS at the 262K target (2026-09-29)

```
GPU KV cache size: 925,447 tokens
Maximum concurrency for 262,144 tokens per request: 3.53x
Model loading took 88.63 GiB      (vs 89.19 with vision — text-only mode saves 0.56 GiB)
init engine (profile, create kv cache, warmup model) took 125.40 s
```

Text-only mode is now confirmed by the engine itself, not inferred:

```
All limits of multimodal modalities supported by the model are set to 0, running in text-only mode.
Disabled mm_prefix attention mode because multimodal inputs are configuration-disabled.
```

and there is **no multi-modal warmup phase at all** — the thing that killed the
previous run.

**Long-context recall verified:** a 60,028-token prompt returned
`The magic word is **"zarquon"**.` Host `MemAvailable` moved 6.6 → 6.1 GB during
that prefill and settled back to 6.4 GB.

### Memory margin is the real constraint, not KV

| | pulsar | magnetar |
|---|---|---|
| idle, model resident @ 262K | **6.6 GB** | 11.1 GB |
| during 60K-token prefill | **6.1 GB** | 10.6 GB |
| earlyoom SIGTERM fires at | ~4.9 GB | ~4.9 GB |

So the working margin on pulsar is **~1.2–1.7 GB**. pulsar consistently runs
~4.5 GB tighter than magnetar and is the binding node. KV is not the limit —
3.53x concurrency at 262K is ample — host memory is. If a future change needs
headroom, `KV_CACHE_MEMORY` is the lever with the most slack.

## Open question: which reasoning parser

**Unresolved — must be probed at rung 1.** The checkpoint card says
`deepseek_r1`; a 2-Spark recipe says `glm45`. A wrong parser does **not** error,
it silently mis-splits `reasoning_content` from `content`, so it can only be
settled by observation.

**Probe three candidates, not two.** This image registers 31 reasoning parsers
including **`glm47`** as well as `glm45` and `deepseek_r1` (verified:
`grep -noE '"(glm[0-9]*|deepseek_r1)"' vllm/reasoning/__init__.py` → lines 23,
55, 59). The branch already concluded that `glm47` is the correct *tool-call*
parser generation for this model and that `glm45` is explicitly wrong there, so
excluding `glm47` from the reasoning probe would be an odd gap — even though the
two parser families are independent. `deepseek_r1` stays the launcher default
because the checkpoint card is the most authoritative single source, but treat
`glm47` as a strong second hypothesis. All three are registered, so none of them
fails at startup; they just quietly disagree about where the thinking goes.

```bash
source glm/.env
probe() {
  curl -s http://localhost:8000/v1/chat/completions \
    -H "Authorization: Bearer ${VLLM_API_KEY}" -H 'Content-Type: application/json' \
    -d '{"model":"zai-org/glm-5.3-flash","messages":[{"role":"user","content":"What is 17*23? Think step by step."}],"max_tokens":512}' \
    | python3 -c 'import json,sys; d=json.load(sys.stdin)["choices"][0]["message"];
print("reasoning_content:", (d.get("reasoning_content") or "<EMPTY>")[:120]);
print("content:", (d.get("content") or "<EMPTY>")[:120])'
}
probe    # once each with REASONING_PARSER=deepseek_r1, =glm47, =glm45
```

Correct parser: `reasoning_content` holds the step-by-step working, `content`
holds just the answer. Wrong parser: `reasoning_content` empty and raw thinking
leaking into `content`, or the reverse.

### Result: `deepseek_r1` LOSES the reasoning (2026-09-29)

Settled by decoding the rendered prompt and generating from it directly.

The chat template ends the prompt with `<|assistant|><think>` (chat_template.jinja
line 256), so **`<think>` is in the prompt, not the generation**. The model then
emits its reasoning and closes with `</think>`. Verified raw:

```
prompt:  '[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>What is 17*23? ...<|assistant|><think>'
output:  "The user wants me to calculate 17 × 23 step by step... Both methods give 391...</think>#"
```

With `--reasoning-parser deepseek_r1` the chat API returns:

* `content` — the post-`</think>` answer, clean, **no leakage** (so this does not
  look like a broken parser from the outside)
* `reasoning_content` — **`None`**, i.e. the entire reasoning block is discarded

So deepseek_r1 splits on `</think>` correctly but never assigns the prefix to
`reasoning_content`, presumably because it expects an opening `<think>` in the
generated text and there is none. This is exactly the silent failure this section
was created to catch: the output looks right, and the reasoning is gone.

**Tested `glm47`: identical to `deepseek_r1`.** `reasoning_content` is `None`,
`content` is clean, no leakage. Also tested `chat_template_kwargs` `{"thinking":
true}` and `{"enable_thinking": true}` — the two keys the adapter actually reads
(`parser/glm47_moe.py:185-186`) — and **streaming**, which produced 0
`reasoning_content` deltas and content deltas starting at `#`.

**Conclusion: in this build, no available parser surfaces the reasoning for this
model.** The parser machinery is behaving as designed — `glm47_moe.py:125` sets
`initial_state=ParserState.REASONING if thinking`, which is why `content` is
always clean — but the REASONING events never reach `reasoning_content` in
either streaming or non-streaming aggregation. That looks like a defect in this
day-0 build's adapter path, not a misconfiguration.

**What this costs:** nothing for ordinary use — `content` is correct and never
contaminated. It costs you the model's chain-of-thought, which is simply
discarded. If you need it, call `/v1/completions` with the rendered prompt (see
the raw-generation transcript above); the reasoning is present there in full.

**Do not spend more restarts on parser names.** The variable that matters is not
which parser, it is that this build drops the events.

Do NOT judge a parser by whether `content` looks clean. Judge it by whether
`reasoning_content` contains the text that appears before `</think>` in the raw
generation.

### Other kwargs worth knowing

The template's knobs are **`reasoning_effort`** (`low` / `high`, default `max`)
and **`clear_thinking`** — *not* `enable_thinking`, which the recipes mention and
which this template ignores entirely.

## Rung 5 notes: CUTLASS MoE (FP4 tensor cores)

marlin is weight-only: it decompresses NVFP4 weights and computes in higher
precision, which is why it logs *"Your GPU does not have native support for FP4
computation"*. That log line is correct rather than a fallback, but it does mean
the GB10 FP4 tensor cores are idle. `flashinfer_cutlass` is the path that uses
them, so rung 5 asks what that is worth against the 14.7 tok/s marlin baseline.

**The first attempt failed, and the reason generalises.** With the nvrtc.h header
link in place the compile starts, but vLLM only triggers it on the first MoE
forward — during KV-cache profiling, with 88.63 GiB of weights already resident
and ~10 GiB of host headroom. The module is **97 nvcc translation units**, and a
single `cicc` on the worst of them measures **5284 MiB RSS**: 4.5x the ~1.17 GiB
`cudafe++` figure that `MAX_JOBS=2` was sized against. earlyoom SIGTERMed the
compiler at object 20 of 97, ~40 minutes into the build on top of a ~9 minute
load, and the worker and engine died with it.

Two fixes, both committed:

1. `run_cluster.sh` now bind-mounts `~/.cache/flashinfer`. Before this the JIT
   output lived in the container's writable layer and was destroyed with the
   container, so every restart paid the build again — the earlier note claiming
   the build was "cached into the bind-mounted cache" was simply wrong; only
   `~/.cache/huggingface` was mounted.
2. `glm/precompile-moe.sh` builds the module with **no model loaded**, where
   ~110 GiB is free instead of ~10 GiB. That inverts the constraint: `MAX_JOBS`
   can go to 10 rather than being throttled to 2.

Run on each node (the JIT cache is per node, like the weights), then restart the
cluster so the containers pick up the new mount:

```bash
bash glm/precompile-moe.sh                      # ~65 min at MAX_JOBS=10
docker stop node-*                              # BOTH nodes
source glm/cluster-env.sh && make head PROFILE=glm      # Node 1
source glm/cluster-env.sh && make worker PROFILE=glm    # Node 2
MAX_MODEL_LEN=262144 MOE_BACKEND=flashinfer_cutlass make serve PROFILE=glm
LABEL=cutlass bash glm/bench.sh
```

A container started **before** the mount existed cannot see the precompiled
module and will try to build it again at profiling time — i.e. it will fail the
same way. The restart is not optional.

Watch for: the engine reaching KV allocation without a compile phase (the module
should load from cache in seconds), and `journalctl -u earlyoom` staying quiet.

## Rung 4 notes: what to watch

**The drafter is 2.34 GB of weights — larger than pulsar's entire 1.2–1.7 GB
margin at 262K.** The recipe's claim that DFlash2 "slot-shares MLA tensors" and
adds no KV cost is about the *KV cache*; the drafter's own parameters are still a
second model loaded on every node. Enabling speculation at the 6 GiB KV budget
walks straight into earlyoom.

So the launcher trades KV down automatically when `ENABLE_DFLASH2=1`: 6 GiB →
3 GiB, which still leaves roughly 460k tokens (~1.75x concurrency at 262K) and
frees 3 GiB — comfortably more than the drafter needs. Override
`KV_CACHE_MEMORY` explicitly to opt out.

Both nodes already hold the drafter (2.2 GB each, fetched 2026-09-29).

Published reference is 46.9 tok/s at 74.1% acceptance, but that came from a
Ray-less rank-launch path, so it may not transfer exactly. Record what this
cluster actually does, including a shortfall.

```bash
curl -s http://localhost:8000/metrics | grep -Ei 'spec_decode|draft|accept'
```

Speculative config in use (from `glm/DISCOVERY.md` — method is `dflash`, not
`dflash2`):

```json
{"method":"dflash","model":"incoai/GLM-5.3-Flash-DFlash2","num_speculative_tokens":7}
```

If rung 4 is stable and materially faster, flip `ENABLE_DFLASH2` to default `1`.
If not, leave it `0` and record why. Either way the drafter stays
CC-BY-NC-ND-4.0 — research use only.
