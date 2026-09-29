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

# Create the unversioned developer symlinks `ld -l<name>` resolves.
#
# The header link above gets the 97 translation units to COMPILE; this gets the
# result to LINK. FlashInfer's final step is
#   c++ ... -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs -lcudart -lcuda -lnvrtc
# and `-lnvrtc` makes ld look for the unversioned `libnvrtc.so`. This image ships
# only the runtime SONAME `libnvrtc.so.13` -- the devel symlink that normally comes
# with the CUDA toolkit package is absent -- so the link dies with
#   /usr/bin/ld: cannot find -lnvrtc: No such file or directory
# after every one of the 97 objects has compiled successfully.
#
# Only nvrtc is affected: libcudart.so is present, and libcuda.so is supplied by
# the stubs directory already on the link line. The loop is written generally and
# guarded so it is a no-op wherever the symlink already exists.
RUN set -eux; \
    for lib in nvrtc; do \
      so="/usr/local/cuda/lib64/lib${lib}.so"; \
      if [ ! -e "$so" ]; then \
        target="$(ls -1 /usr/local/cuda/lib64/lib${lib}.so.* 2>/dev/null | head -1 || true)"; \
        if [ -n "$target" ]; then ln -s "$(basename "$target")" "$so"; fi; \
      fi; \
    done; \
    test -e /usr/local/cuda/lib64/libnvrtc.so
