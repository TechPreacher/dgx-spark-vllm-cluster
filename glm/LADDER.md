# GLM-5.3-Flash context ladder — measurement log

Status: **prepared, not yet run.** Blocked on the host-hardening prerequisites
below, which need `sudo`. Fill each rung in as it is climbed; do not skip rungs.

Headroom here is ~12.9 GiB/node against Nemotron's roughly double, and this
cluster has a documented memory-starvation failure (`gpt-oss-120b`) that took
sshd unreachable while ICMP still replied and needed a power cycle to recover.
That is why the ladder exists and why the prerequisites are prerequisites.

## Prerequisites — run on BOTH nodes before rung 1

As of 2026-09-28 **none of the first three are satisfied on either node**:

| Check | pulsar | magnetar | Required |
|---|---|---|---|
| `earlyoom` | not installed | not installed | installed + enabled |
| sshd `OOMScoreAdjust` | `0` | `0` | `-1000` |
| `vm.swappiness` | `60` | `60` | `0` |
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

Also required: `glm/.env` with a real `HF_TOKEN` whose account has accepted terms
for `LibertAIDAI/GLM-5.3-Flash-NVFP4`, and for `incoai/GLM-5.3-Flash-DFlash2` if
rung 4 is attempted.

First run downloads ~181 GiB into `~/.cache/huggingface` (381 GB already used,
3.1 TB free — fits).

## Rungs

| Rung | ctx | util | KV | Spec | Proves | MemAvail pulsar | MemAvail magnetar | decode tok/s | Result |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 32K | 0.80 | fp8 | off | Weights load; TP2 collectives alive across RoCE | | | | _pending_ |
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

Result: _pending_. Once known, set it as the default in
`glm/launch-glm53-flash.sh` and record the evidence here.

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
