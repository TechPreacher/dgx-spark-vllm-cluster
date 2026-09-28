# Source this file BEFORE bringing up the Ray cluster when you intend to run
# LibertAIDAI/GLM-5.3-Flash-NVFP4. These vars must be present in the Ray
# container env at START time on BOTH nodes -- Ray cannot propagate them from
# the head driver across nodes to worker ranks at vllm-serve time, so a var set
# only on the head gives rank 1 a different backend and the run hangs in a
# collective rather than failing with a message.
#
# Usage:
#   # On Node 1 (head):
#   source glm/cluster-env.sh
#   make head PROFILE=glm
#
#   # On Node 2 (worker):
#   source glm/cluster-env.sh
#   make worker PROFILE=glm
#
#   # Then on Node 1 in a new terminal:
#   make serve PROFILE=glm
#
# Source order matters: run_*node_2.sh expands VLLM_FORWARD_VARS at script
# start, so these must already be exported in the parent shell.
#
# Deliberately NOT set here: NCCL_IB_HCA, NCCL_SOCKET_IFNAME, UCX_NET_DEVICES.
# select_up_dataplane (cluster/lib.sh) builds those at launch from the CX7 links
# that currently have carrier -- all 4 after a cold boot, only the 2 f0 halves
# after a warm reboot. A static value here would hand NCCL a down HCA after a
# warm reboot and stall collective init.

export VLLM_IMAGE=local/vllm-ray-glm53:sm121-v11-dflash2

# ---------------------------------------------------------------------------
# Why this list is SHORT, and deliberately not a copy of nemotron's
# ---------------------------------------------------------------------------
# The Nemotron profile forwards four NVFP4 vars. Two of them DO NOT EXIST in
# this image's vLLM build -- verified by grepping its envs.py:
#
#   VLLM_NVFP4_GEMM_BACKEND      absent (nemotron sets it; no-op here)
#   VLLM_USE_FLASHINFER_MOE_FP4  absent (only ..._MOE_INT4 exists here)
#   VLLM_FLASHINFER_ALLREDUCE_BACKEND  present, left at the image's default
#   VLLM_ALLOW_LONG_MAX_MODEL_LEN      present, forwarded below
#
# This is a different vLLM (the patched glm53-flash build, vllm
# 0.1.dev20051+g487ecf187) from the NGC 26.05 image the Nemotron path uses, so
# its env surface differs. Setting vars that do not exist would be harmless but
# actively misleading to the next reader. VLLM_ATTENTION_BACKEND is likewise not
# env-selectable in this build, so the SM121 NoPE-MLA/FA2 path is chosen by the
# image's own patches rather than by us.
#
# The image's tuned defaults are left alone on purpose: it exists specifically
# to make SM121 work, so overriding its backend choices is the opposite of what
# we want. See glm/DISCOVERY.md for how each of these was established.
#
# VLLM_ALLOW_LONG_MAX_MODEL_LEN is not strictly required at 262144 (below the
# checkpoint's native maximum) but is forwarded as cheap insurance: the moment
# anyone raises MAX_MODEL_LEN toward the native 1M it IS required, and the
# failure without it on rank 1 only is a hang, not an error.
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# Tells run_*node_2.sh which variables to forward into the Ray container's env.
# Append additional names here if you add more model-specific runtime flags.
export VLLM_FORWARD_VARS="VLLM_ALLOW_LONG_MAX_MODEL_LEN"
