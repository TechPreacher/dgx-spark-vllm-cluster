.PHONY: help head worker nemotron check-nvidia fix-nvidia preboot-check preflight

SHELL := /bin/bash

help:
	@echo "Targets:"
	@echo "  make head          - start Ray head node (run on 10.0.1.3)"
	@echo "  make worker        - start Ray worker node (run on Node 2)"
	@echo "  make nemotron      - launch Nemotron-3-Super-120B (run on head after both nodes up)"
	@echo
	@echo "  make check-nvidia  - check the NVIDIA driver on THIS node (run on each Spark)"
	@echo "  make preboot-check - BEFORE rebooting THIS node: will it come up with a GPU?"
	@echo "  make fix-nvidia    - install NVIDIA modules matching this node's kernel"
	@echo
	@echo "  head/worker run check-nvidia first; bypass with SKIP_PREFLIGHT=1"

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
			echo "preflight: aborting bring-up. Bypass with: make $(MAKECMDGOALS) SKIP_PREFLIGHT=1" >&2; \
			exit 1; \
		fi; \
	fi

head: preflight
	cd cluster/head && . ../../nemotron/cluster-env.sh && bash run_headnode_2.sh

worker: preflight
	cd cluster/worker && . ../../nemotron/cluster-env.sh && bash run_workernode_2.sh

nemotron:
	cd nemotron && bash launch-nemotron-120b.sh
