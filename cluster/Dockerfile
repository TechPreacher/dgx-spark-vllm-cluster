# Derive a Ray-capable vLLM image from the NGC base.
#
# The nvcr.io/nvidia/vllm:26.05* images dropped Ray from their default install
# (verified via `pip show ray` returning not-found), but the Ray-based TP=2
# topology this repo uses needs the `ray` console script + Python module on
# every node. We add it back on top of the NGC base so we keep the newer vLLM
# features (super_v3 reasoning parser plugin, --async-scheduling, --moe-backend,
# --max-cudagraph-capture-size, --mamba-ssm-cache-dtype float16, NVFP4 kernels).
#
# Build with cluster/build-image.sh (see that script for tag conventions).
# Re-run the build on each node any time you bump BASE_IMAGE.

ARG BASE_IMAGE=nvcr.io/nvidia/vllm:26.05.post1-py3
FROM ${BASE_IMAGE}

# `ray[default]` pulls the dashboard + observability extras the Ray CLI expects.
# Not pinned: leave pip free to pick a version compatible with this vLLM build.
# If you hit a Ray/vLLM mismatch, pin here (e.g. "ray[default]==2.40.0") and rebuild.
RUN pip install --no-cache-dir "ray[default]" \
 && ray --version

# Expose nvrtc.h where nvcc looks for it.
#
# FlashInfer JIT-builds its CUTLASS fused-MoE module at first use, and that build
# includes <nvrtc.h> via tensorrt_llm/deep_gemm/jit_utils.cuh. The header is not
# in /usr/local/cuda/include in these images -- it ships only inside the pip
# wheel at dist-packages/nvidia/cu*/include/ -- and FlashInfer's nvcc command
# line does not add that directory, so the build dies with
#   jit_utils.cuh:21:10: fatal error: nvrtc.h: No such file or directory
# ~9 minutes into a model load, at KV-cache profiling.
#
# Link every header from the wheel that CUDA's include dir is missing. Guarded so
# it is a no-op on images that already have them (e.g. the NGC base).
RUN set -eux; \
    for d in /usr/local/lib/python3.*/dist-packages/nvidia/cu*/include; do \
      [ -d "$d" ] || continue; \
      for h in "$d"/*.h; do \
        [ -e "$h" ] || continue; \
        t="/usr/local/cuda/include/$(basename "$h")"; \
        [ -e "$t" ] || ln -s "$h" "$t"; \
      done; \
    done; \
    test -e /usr/local/cuda/include/nvrtc.h
