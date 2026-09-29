# GLM-5.3-Flash context ladder — measurement log

Status: **all prerequisites verified green on both nodes 2026-09-28; rungs not
yet run.** The only remaining blocker for rung 1 is `glm/.env`. Fill each rung in
as it is climbed; do not skip rungs.

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
| 2 | 131K | 0.85 | fp8, 6 GiB | off | KV math holds | | | | _pending_ |
| 3 | 262K | 0.85 | fp8, 6 GiB | off | Target context | | | | _pending_ |
| 4 | 262K | 0.85 | fp8, 6 GiB | dflash, 7 | Acceptance + tok/s vs published 46.9 / 74.1% | | | | _pending_ |

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

**Next:** re-test with `REASONING_PARSER=glm47`, then `glm45`. glm47 is the
generation that matches this model's tool parser, so it is the leading candidate.
Each change needs a server restart (~13 min).

Do NOT judge a parser by whether `content` looks clean. Judge it by whether
`reasoning_content` contains the text that appears before `</think>` in the raw
generation.

### Other kwargs worth knowing

The template's knobs are **`reasoning_effort`** (`low` / `high`, default `max`)
and **`clear_thinking`** — *not* `enable_thinking`, which the recipes mention and
which this template ignores entirely.

## Rung 4 notes: what to watch

The drafter slot-shares MLA tensors and should add **no** KV cost, so a large
`MemAvailable` drop when enabling it is a signal something is wrong, not a cost
to accept.

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
