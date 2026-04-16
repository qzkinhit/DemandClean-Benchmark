#!/bin/bash
# =============================================================
# DemandClean-Benchmark — One-Click Runner
#
# Delegates to run_demandclean/run.sh with all arguments.
#
# Usage:
#   bash run.sh                                   # default: 300 episodes, all datasets
#   bash run.sh --dataset beers --n_episodes 50   # specific dataset + episodes
#   bash run.sh --dataset beers --versions v5     # specific version
#   bash run.sh --all_datasets --n_episodes 300   # all datasets
#
# See run_demandclean/run.sh for full usage details.
# =============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_demandclean/run.sh" "$@"
