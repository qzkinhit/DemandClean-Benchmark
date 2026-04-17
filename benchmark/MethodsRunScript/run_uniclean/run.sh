#!/bin/bash
# ============================================================
# UniClean Baseline full benchmark script
# Prerequisites:
# - Install JDK 8 or 11
# - Set JAVA_HOME
# - Install PySpark
# Supports --dataset argument and VERSION env var
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

VERSION="${VERSION:-}"

# Parse command-line arguments
SELECTED_DATASET=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) SELECTED_DATASET="$2"; shift 2 ;;
        --version) VERSION="$2"; shift 2 ;;
        *) shift ;;
    esac
done

echo "============================================================"
echo "UniClean Baseline full benchmark"
echo "Project root: $PROJECT_ROOT"
echo "Version tag: ${VERSION:-none}"
echo "Selected dataset: ${SELECTED_DATASET:-all}"
echo "Start time: $(date)"
echo "============================================================"

# Check Java environment
if [ -z "$JAVA_HOME" ]; then
    echo "Warning: JAVA_HOME is not set; UniClean requires a Java environment"
fi

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate multibaseline

mkdir -p logs/uniclean
mkdir -p results/uniclean

declare -a ALL_DATASETS=(
    "adult|income|classification"
#    "beers|style|classification"
#    "bike|cnt|regression"
#    "breast_cancer|class|classification"
#    "har|gt|clustering"
    "mercedes|y|regression"
#    "nasa|sound_pressure_level|regression"
    "smartfactory|labels|classification"
#    "soilmoisture|soil_moisture|regression"
)

MODELS_CLASSIFICATION="rf lr svm knn dt gb"
MODELS_REGRESSION="rf lr ridge lasso knn gb"
MODELS_CLUSTERING="kmeans agglomerative"

# UniClean parameters
SINGLE_MAX=10000
EXECUTOR_MEMORY="48g"
DRIVER_MEMORY="48g"

total=0; success=0; failed=0

for dataset_config in "${ALL_DATASETS[@]}"; do
    IFS='|' read -r dataset label_column task_type <<< "$dataset_config"

    if [[ -n "$SELECTED_DATASET" && "$dataset" != "$SELECTED_DATASET" ]]; then
        continue
    fi

    echo "------------------------------------------------------------"
    echo "Dataset: $dataset | label: $label_column | task: $task_type"
    echo "------------------------------------------------------------"

    case $task_type in
        "classification") models=$MODELS_CLASSIFICATION ;;
        "regression") models=$MODELS_REGRESSION ;;
        "clustering") models=$MODELS_CLUSTERING ;;
        *) models=$MODELS_CLASSIFICATION ;;
    esac

    dirty_path="Data/${dataset}/dirty_index.csv"
    clean_path="Data/${dataset}/clean_index.csv"
    task_name="${dataset}_uniclean${VERSION:+_$VERSION}"
    log_file="logs/uniclean/${task_name}.log"

    [[ ! -f "$dirty_path" ]] && echo "skip: $dirty_path does not exist" && failed=$((failed + 1)) && continue
    [[ ! -f "$clean_path" ]] && echo "skip: $clean_path does not exist" && failed=$((failed + 1)) && continue

    cmd="python MethodsRunScript/run_uniclean/run_uniclean_base.py \
        --dirty_path $dirty_path --clean_path $clean_path \
        --dataset $dataset \
        --task_name $task_name --output_path results/uniclean/ \
        --index_attribute index --label_column $label_column \
        --task_type $task_type --models $models \
        --single_max $SINGLE_MAX \
        --executor_memory $EXECUTOR_MEMORY --driver_memory $DRIVER_MEMORY --verbose"

    echo "log: $log_file"
    total=$((total + 1))
    if eval "$cmd" 2>&1 | tee "$log_file"; then
        echo "[OK] success: $dataset"; success=$((success + 1))
    else
        echo "[FAIL] failure: $dataset"; failed=$((failed + 1))
    fi
done

echo "============================================================"
echo "UniClean benchmark done: total=$total success=$success failure=$failed"
echo "============================================================"
