# Source this file BEFORE bringing up the Ray cluster when you intend to run
# nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4. These vars must be present in
# the Ray container env at start time on BOTH nodes -- Ray cannot propagate
# them from the head driver across nodes to worker ranks at vllm-serve time.
#
# Usage:
#   # On Node 1 (head):
#   source nemotron/cluster-env.sh
#   cd cluster/head && bash run_headnode_2.sh
#
#   # On Node 2 (worker):
#   source nemotron/cluster-env.sh
#   cd cluster/worker && bash run_workernode_2.sh
#
#   # Then on Node 1 in a new terminal:
#   cd nemotron && ./launch-nemotron-120b.sh
#
# Source order matters: run_*node_2.sh expands VLLM_FORWARD_VARS at script start,
# so the four NVFP4 vars must already be exported in the parent shell.

# Pin this profile's image explicitly. Do NOT rely on run_*node_2.sh's
# ${VLLM_IMAGE:-local/vllm-ray:26.05.post1} default: glm/cluster-env.sh exports
# VLLM_IMAGE, so in a shell that sourced the glm profile first -- exactly what
# happens when an operator Ctrl-Cs a blocking bring-up to switch models -- the
# leaked GLM image would be inherited here and the Nemotron launcher has no
# image guard to catch it. Worse, a fresh shell on the worker would default
# correctly, leaving the two ranks on different vLLM AND Ray builds.
export VLLM_IMAGE=local/vllm-ray:26.05.post1

export VLLM_NVFP4_GEMM_BACKEND=marlin
export VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm
export VLLM_USE_FLASHINFER_MOE_FP4=0
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# Tells run_*node_2.sh which variables to forward into the Ray container's env.
# Append additional names here if you add more model-specific runtime flags.
export VLLM_FORWARD_VARS="VLLM_NVFP4_GEMM_BACKEND VLLM_FLASHINFER_ALLREDUCE_BACKEND VLLM_USE_FLASHINFER_MOE_FP4 VLLM_ALLOW_LONG_MAX_MODEL_LEN"
