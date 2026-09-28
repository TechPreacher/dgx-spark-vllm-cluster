.PHONY: help head worker serve require-profile check-nvidia fix-nvidia preboot-check preflight test verify-glm-image

SHELL := /bin/bash

help:
	@echo "Targets (PROFILE is REQUIRED on head/worker/serve -- no default):"
	@echo "  make head   PROFILE=<$(PROFILE_LIST)>  - start Ray head node (run on Node 1)"
	@echo "  make worker PROFILE=<$(PROFILE_LIST)>  - start Ray worker node (run on Node 2)"
	@echo "  make serve  PROFILE=<$(PROFILE_LIST)>  - launch the model (on head, after both nodes up)"
	@echo
	@echo "  make check-nvidia  - check the NVIDIA driver on THIS node (run on each Spark)"
	@echo "  make preboot-check - BEFORE rebooting THIS node: will it come up with a GPU?"
	@echo "  make fix-nvidia    - install NVIDIA modules matching this node's kernel"
	@echo "  make test          - run the repo's shell unit tests"
	@echo "  make verify-glm-image - gate the GLM image: pinned base digest + unmoved pins"
	@echo
	@echo "  head/worker run check-nvidia first; bypass with SKIP_PREFLIGHT=1"
	@echo
	@echo "  PROFILE has no default on purpose: the profiles carry different"
	@echo "  VLLM_FORWARD_VARS and different container images, and a wrong-image"
	@echo "  bring-up hangs in an NCCL collective rather than erroring."

test:
	@bash scripts/test_nvidia_lib.sh
	@bash scripts/test_cluster_lib.sh

check-nvidia:
	@bash scripts/check_nvidia.sh

preboot-check:
	@bash scripts/preboot_check.sh

fix-nvidia:
	@bash scripts/fix_nvidia.sh

# Gate the cluster bring-up on the host driver actually working. Without this a
# missing nvidia.ko surfaces only as an opaque docker failure:
#   "nvidia-container-cli: initialization error: nvml error: driver not loaded"
#
# Blocks on exit 1 (driver unusable) and 2 (check could not run). Exit 3 is
# metapackage drift -- a next-reboot risk, not a today problem -- so it warns
# and lets the bring-up proceed.
preflight:
	@if [ "$(SKIP_PREFLIGHT)" = "1" ]; then \
		echo "preflight: skipped (SKIP_PREFLIGHT=1)"; \
	else \
		bash scripts/check_nvidia.sh; rc=$$?; \
		if [ $$rc -eq 1 ] || [ $$rc -eq 2 ]; then \
			echo "preflight: aborting bring-up. Bypass with: make $(MAKECMDGOALS) PROFILE=$(PROFILE) SKIP_PREFLIGHT=1" >&2; \
			exit 1; \
		fi; \
	fi

# PROFILE selects the model profile: its cluster-env.sh supplies both the
# VLLM_FORWARD_VARS set and VLLM_IMAGE. There is deliberately NO default.
#
# The two profiles need *different* forwarded vars and different images, so a
# default would let a bare `make head` bring the cluster up on the wrong image
# with the wrong env -- which does not error, it hangs in a collective when
# rank 1 picks a different backend from rank 0. An explicit profile on every
# invocation is cheap; diagnosing a silent wrong-image bring-up is not.
VALID_PROFILES := nemotron glm
PROFILE_LIST := $(shell echo '$(VALID_PROFILES)' | tr ' ' '|')

require-profile:
	@if [ -z "$(PROFILE)" ]; then \
		echo "PROFILE is required. Valid profiles: $(VALID_PROFILES)" >&2; \
		echo "  e.g.  make $(firstword $(MAKECMDGOALS)) PROFILE=glm" >&2; \
		exit 1; \
	fi
	@if ! echo "$(VALID_PROFILES)" | tr ' ' '\n' | grep -qx "$(PROFILE)"; then \
		echo "Unknown PROFILE '$(PROFILE)'. Valid profiles: $(VALID_PROFILES)" >&2; \
		exit 1; \
	fi
	@if [ ! -r "$(PROFILE)/cluster-env.sh" ]; then \
		echo "PROFILE '$(PROFILE)' has no $(PROFILE)/cluster-env.sh" >&2; \
		exit 1; \
	fi

# `unset VLLM_IMAGE` first: a profile that does not pin it would otherwise
# inherit a leaked value from whatever profile the operator's shell sourced
# earlier, and bring the cluster up on the wrong image with no error.
head: require-profile preflight
	cd cluster/head && unset VLLM_IMAGE && . ../../$(PROFILE)/cluster-env.sh && bash run_headnode_2.sh

worker: require-profile preflight
	cd cluster/worker && unset VLLM_IMAGE && . ../../$(PROFILE)/cluster-env.sh && bash run_workernode_2.sh

serve: require-profile
	@case "$(PROFILE)" in \
	  nemotron) cd nemotron && bash launch-nemotron-120b.sh ;; \
	  glm)      cd glm && bash launch-glm53-flash.sh ;; \
	  *) echo "No serve wiring for PROFILE=$(PROFILE). Add it to the serve target." >&2; exit 1 ;; \
	esac

verify-glm-image:
	@bash glm/verify-image.sh
