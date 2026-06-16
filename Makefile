.PHONY: help head worker nemotron

SHELL := /bin/bash

help:
	@echo "Targets:"
	@echo "  make head      - start Ray head node (run on 10.0.1.3)"
	@echo "  make worker    - start Ray worker node (run on Node 2)"
	@echo "  make nemotron  - launch Nemotron-3-Super-120B (run on head after both nodes up)"

head:
	cd cluster/head && . ../../nemotron/cluster-env.sh && bash run_headnode_2.sh

worker:
	cd cluster/worker && . ../../nemotron/cluster-env.sh && bash run_workernode_2.sh

nemotron:
	cd nemotron && bash launch-nemotron-120b.sh
