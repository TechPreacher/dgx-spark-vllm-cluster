# GLM-5.3-Flash: what was read off the image, and how

Everything here was established by inspecting the artifacts on 2026-09-28, not
taken from prose recipes. Each finding carries the command that produced it, so
it can be re-checked after an image bump. Several published claims did **not**
survive contact with the image — those are called out.

Image inspected: `local/vllm-ray-glm53:sm121-v11-dflash2`, built from
`ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6`
(arm64/linux, 14.2 GB compressed, 48 layers, anonymously pullable).

## The pins that matter

```bash
docker run --rm --entrypoint /bin/bash <img> -c \
  'pip freeze | grep -iE "^(torch|flashinfer|nvidia-nccl|nvidia-cutlass|vllm|ray)" | sort'
```

| Package | Version |
|---|---|
| `flashinfer-python` / `flashinfer-cubin` | `0.6.18.dev20260819` |
| `nvidia-cutlass-dsl` | `4.6.2` |
| `nvidia-nccl-cu13` | `2.30.7` |
| `torch` | `2.13.0+cu130` |
| `vllm` | `0.1.dev20051+g487ecf187` (local aarch64 wheel) |
| `ray` | `2.58.0` (added by our layer) |

These match the recipe's claims. FlashInfer 0.6.18 is the load-bearing one:
0.6.17 produced NaN on SM121 at batch 64–256 rows. `glm/verify-image.sh` gates
on none of these moving.

## GPU works under our driver

```bash
docker run --rm --gpus all --entrypoint /bin/bash <img> -c \
  'nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; \
   python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_capability())"'
# NVIDIA GB10, 580.178.04
# True (12, 1)
```

Compute capability `(12, 1)` = SM121, confirmed at runtime. Note the image's
`NVIDIA_REQUIRE_CUDA` label enumerates driver ranges only up to `575,<576`,
which does **not** include our 580.178.04 — but the container starts and sees
the GPU anyway on this host, so the constraint is not enforced here. Worth
remembering if a future toolkit upgrade starts honouring it: the escape hatch is
`NVIDIA_DISABLE_REQUIRE=1`.

Also note `TORCH_CUDA_ARCH_LIST=8.0 8.7 8.9 9.0 10.0 11.0 12.0` — it lists 12.0,
not 12.1. Kernels run on SM121 via minor-version compatibility. Not a problem in
practice, but it is why this is a patched image and not a stock one.

## There is NO launcher script in the image

```bash
docker run --rm --entrypoint /bin/bash <img> -c \
  'find / -xdev \( -iname "*launch*glm*" -o -iname "*dflash*" -o -iname "*glm53*" \) 2>/dev/null'
```

Returns only Python modules inside vLLM. The recipe's
`launch-glm53-vllm-tp2-dflash2.sh` lives in its GitHub repo, not in the
published image. So the plan's "read the shipped launcher for the serve flags"
was not possible; the flags below were derived from vLLM's own code and the
checkpoints' configs instead.

## Model architecture and DFlash2 wiring

```bash
docker run --rm --entrypoint /bin/bash <img> -c \
  'grep -nE "Glm5Next|DFlash" /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/registry.py'
```

| Registry key | Implementation |
|---|---|
| `Glm5NextForCausalLM` | `vllm.models.glm5next` |
| `Glm5NextForConditionalGeneration` | `vllm.models.glm5next` (multimodal) |
| `Glm5NextMTPModel` | `vllm.models.glm5next` → `Glm5NextMTP` |
| `DFlash2DraftModel` | `qwen3_dflash2` → `DFlash2Qwen3ForCausalLM` |

The target checkpoint `LibertAIDAI/GLM-5.3-Flash-NVFP4` declares
`architectures: ["Glm5NextForConditionalGeneration"]` — the **multimodal**
variant, with `quant_method: modelopt` and 43 excluded modules (the BF16
attention / vision tower / shared experts / router / embeddings / LM head).
That is why `--skip-mm-profiling` matters even for a text-only run: the vision
tower is present in the graph regardless.

`vllm/models/glm5next/nvidia/model.py` contains an explicit
`DFLASH2-AUX-CAPTURE` block (EAGLE-3 style aux hidden states), so DFlash2 is
genuinely wired for this model in this build — the target captures the hidden
states the drafter consumes.

## The DFlash2 drafter

Official checkpoint: **`incoai/GLM-5.3-Flash-DFlash2`**

```bash
curl -fsSL https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/raw/main/config.json
curl -fsSL https://huggingface.co/api/models/incoai/GLM-5.3-Flash-DFlash2
```

- `architectures: ["DFlash2DraftModel"]` → resolves to the **Qwen3** DFlash2
  class. The drafter is Qwen3-shaped (HF tags include `qwen3`), which is why no
  GLM-specific DFlash2 class exists and none is needed.
- `dflash_config`: `block_size: 8`, `conv_kernel_size: 2`, `conv_group_size: 16`,
  `selector_rank: 256`, `selector_top_k: 16`,
  `target_layer_ids: [5, 14, 24, 33, 42]`, `mask_token_id: 154856`
- `hidden_size: 4096`, `intermediate_size: 12288`, `head_dim: 128`,
  `layer_types`: all `sliding_attention`, `is_causal: false` (block diffusion)
- Single `model.safetensors`
- **Licence: `cc-by-nc-nd-4.0`** — confirmed. Research/personal use only, no
  redistribution, never baked into a shared image.

## The speculative method is `dflash`, not `dflash2`

```bash
docker run --rm --entrypoint /bin/bash <img> -c \
  'grep -nE "DFlashModelTypes|def use_dflash" /usr/local/lib/python3.12/dist-packages/vllm/config/speculative.py'
# DFlashModelTypes = Literal["dflash"]
# def use_dflash(self): return self.method == "dflash"
```

So the flag is `--speculative-config` with `"method": "dflash"`. `"dflash2"` is
not an accepted method string — the "2" lives in the drafter's architecture, not
in the method name. vLLM also derives `n_predict` from the drafter's
`block_size` automatically when unset, and sets `parallel_drafting = True` for
`dflash`.

**`GLM_SPEC_CONFIG` for Task 7:**

```
--speculative-config {"method":"dflash","model":"incoai/GLM-5.3-Flash-DFlash2","num_speculative_tokens":7}
```

`num_speculative_tokens: 7` against the drafter's `block_size: 8` = a block of 8
including the bonus token, which matches the recipe's `--num-speculative-tokens 7`.

## VLLM_FORWARD_VARS: why the list is one entry

```bash
docker run --rm --entrypoint /bin/bash <img> -c \
  'grep -oE "\"VLLM_(NVFP4[A-Z0-9_]*|USE_FLASHINFER[A-Z0-9_]*|FLASHINFER[A-Z0-9_]*|ATTENTION_BACKEND|ALLOW_LONG_MAX_MODEL_LEN|MLA[A-Z0-9_]*)\"" \
     /usr/local/lib/python3.12/dist-packages/vllm/envs.py | tr -d \" | sort -u'
```

| Var | In this build? | Decision |
|---|---|---|
| `VLLM_NVFP4_GEMM_BACKEND` | **absent** | not forwarded (Nemotron sets it; would be a no-op here) |
| `VLLM_USE_FLASHINFER_MOE_FP4` | **absent** (only `..._MOE_INT4`) | not forwarded |
| `VLLM_ATTENTION_BACKEND` | **absent** | not settable; the image's patches pick the SM121 NoPE-MLA/FA2 path |
| `VLLM_FLASHINFER_ALLREDUCE_BACKEND` | present | left at the image's default |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | present | **forwarded** |

This is the discovery that most contradicts the plan: the GLM profile needs a
*different and much smaller* var set than Nemotron, because the patched
glm53-flash vLLM is a different build from NGC 26.05 with a different env
surface. Copying Nemotron's four vars would have set two that do not exist.

Neither `vllm/models/glm5next/` nor the dflash spec-decode path references any
`VLLM_*` var at all:

```bash
docker run --rm --entrypoint /bin/bash <img> -c \
  'grep -rhoE "VLLM_[A-Z0-9_]+" /usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/ | sort -u'
# (no output)
```

`VLLM_ALLOW_LONG_MAX_MODEL_LEN` is not strictly needed at 262144, which is below
the checkpoint's native maximum. It is forwarded as cheap insurance: it becomes
required the moment `MAX_MODEL_LEN` is raised toward the native 1M, and its
absence on rank 1 only is a hang rather than an error.

`NCCL_IB_ADDR_RANGE` from the recipe is **not** set: this repo's bring-up derives
the data plane from carrier-up links via `select_up_dataplane`, and the existing
Qwen and Nemotron TP=2 paths work without it. Pinning addressing statically would
also break the warm-reboot case, where only the two `f0` halves have carrier.
