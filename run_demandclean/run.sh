#!/bin/bash
# ============================================================
# DemandClean Evaluation Runner
#
# One-click script for DemandClean training, inference, and evaluation.
# Logs:    logs/demandclean/
# Results: results/demandclean/{dataset}/{version}/
#
# Usage:
#   bash run_demandclean/run.sh                                  # default: 300 episodes, all datasets
#   bash run_demandclean/run.sh --n_episodes 10                  # quick test with 10 episodes
#   bash run_demandclean/run.sh --dataset beers --n_episodes 50  # specific dataset + episodes
#   bash run_demandclean/run.sh --dataset beers --versions v3    # specific version
#   bash run_demandclean/run.sh --dataset beers --versions v1,v3,v5  # multiple versions
#   bash run_demandclean/run.sh --all_datasets --n_episodes 300  # all datasets
#
#   # Custom error injection rates
#   bash run_demandclean/run.sh --dataset beers --missing_rate 0.05,0.1 \
#       --semantic_rate 0.1,0.2 --syntactic_rate 0.15,0.3 --label_rate 0.0,0.05
#
# Version descriptions (v1-v8):
#   v1: oracle + dueling + single_phase
#   v2: oracle + dueling + two_phase
#   v3: oracle + plain   + single_phase
#   v4: oracle + plain   + two_phase
#   v5: auto   + dueling + single_phase (default)
#   v6: auto   + dueling + two_phase
#   v7: auto   + plain   + single_phase
#   v8: auto   + plain   + two_phase
#
# Supported datasets:
#   beers, adult, bike, breast_cancer, har, mercedes, nasa, smartfactory, soilmoisture
# ============================================================

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Ensure unbuffered Python output (real-time log writing)
export PYTHONUNBUFFERED=1

# Python interpreter
PYTHON="python3"

echo "============================================================"
echo "DemandClean Evaluation"
echo "Project root: $PROJECT_ROOT"
echo "Python: $PYTHON"
echo "Arguments: $@"
echo "Start time: $(date)"
echo "============================================================"

mkdir -p logs/demandclean

# Pass all arguments through to the Python script
$PYTHON run_demandclean/run_demandclean_base.py "$@"
exit_code=$?

# Generate summary report on success
if [ $exit_code -eq 0 ]; then
    echo "------------------------------------------------------------"
    echo "Generating experiment summary report..."
    $PYTHON run_demandclean/generate_report.py "$@"
    report_code=$?
    if [ $report_code -eq 0 ]; then
        echo "Summary report generated successfully"
    else
        echo "[WARNING] Summary report generation failed (exit code: $report_code)"
    fi
fi

echo "============================================================"
echo "DemandClean evaluation complete"
echo "Exit code: $exit_code"
echo "End time: $(date)"
echo "Results: results/demandclean/"
echo "Logs:    logs/demandclean/"
echo "============================================================"

exit $exit_code
