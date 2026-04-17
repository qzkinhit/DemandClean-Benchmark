#!/bin/bash
# ============================================================
# DoNothing Baseline full benchmark script
#
# Purpose: run the benchmark on all datasets in one go
# Supports --dataset <name> to select a single dataset
# Supports VERSION env var to append a version suffix
# ============================================================

set -e

trap 'echo "Script failed at line $LINENO , exit code: $?"' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Read version tag (passed by run_all.sh or set manually)
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
echo "DoNothing Baseline full benchmark"
echo "Project root: $PROJECT_ROOT"
echo "Version tag: ${VERSION:-none}"
echo "Selected dataset: ${SELECTED_DATASET:-all}"
echo "Start time: $(date)"
echo "============================================================"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate multibaseline

mkdir -p logs/donothing
mkdir -p results/donothing

# Dataset configuration: dataset|label_column|task_type|mse_attrs
declare -a ALL_DATASETS=(
    "adult|income|classification|age,fnlwgt,capital_gain,capital_loss,hours_per_week"
    "beers|style|classification|ibu,abv"
    "bike|cnt|regression|temp,atemp,hum,windspeed,cnt"
    "breast_cancer|class|classification|"
    "har|gt|clustering|"
    "mercedes|y|regression|y"
    "nasa|sound_pressure_level|regression|frequency,angle,chord_length,velocity,thickness,sound_pressure_level"
    "smartfactory|labels|classification|o_w_blo_power,o_w_blo_voltage,o_w_bhl_power,o_w_bhl_voltage"
    "soilmoisture|soil_moisture|regression|soil_moisture,soil_temperature"
)

MODELS_CLASSIFICATION="rf lr svm knn dt gb"
MODELS_REGRESSION="rf lr ridge lasso knn gb"
MODELS_CLUSTERING="kmeans agglomerative"

total=0
success=0
failed=0

for dataset_config in "${ALL_DATASETS[@]}"; do
    IFS='|' read -r dataset label_column task_type mse_attrs <<< "$dataset_config"

    # If a specific dataset was requested, skip all others
    if [[ -n "$SELECTED_DATASET" && "$dataset" != "$SELECTED_DATASET" ]]; then
        continue
    fi

    echo "------------------------------------------------------------"
    echo "Dataset: $dataset"
    echo "Label column: $label_column"
    echo "Task type: $task_type"
    echo "------------------------------------------------------------"

    case $task_type in
        "classification") models=$MODELS_CLASSIFICATION ;;
        "regression") models=$MODELS_REGRESSION ;;
        "clustering") models=$MODELS_CLUSTERING ;;
        *) models=$MODELS_CLASSIFICATION ;;
    esac

    dirty_path="Data/${dataset}/dirty_index.csv"
    clean_path="Data/${dataset}/clean_index.csv"
    task_name="${dataset}_donothing${VERSION:+_$VERSION}"
    log_file="logs/donothing/${task_name}.log"

    if [[ ! -f "$dirty_path" ]]; then
        echo "Warning: dirty data file does not exist: $dirty_path, skipping..."
        failed=$((failed + 1))
        continue
    fi

    if [[ ! -f "$clean_path" ]]; then
        echo "Warning: clean data file does not exist: $clean_path, skipping..."
        failed=$((failed + 1))
        continue
    fi

    cmd="python MethodsRunScript/run_donothing/run_donothing_base.py \
        --dirty_path $dirty_path \
        --clean_path $clean_path \
        --task_name $task_name \
        --output_path results/donothing/ \
        --index_attribute index \
        --label_column $label_column \
        --task_type $task_type \
        --models $models \
        --verbose"

    if [[ -n "$mse_attrs" ]]; then
        mse_attrs_space=$(echo "$mse_attrs" | tr ',' ' ')
        cmd="$cmd --mse_attributes $mse_attrs_space"
    fi

    echo "Command: $cmd"
    echo "Log file: $log_file"

    total=$((total + 1))
    if eval "$cmd" 2>&1 | tee "$log_file"; then
        echo "[OK] success: $dataset"
        success=$((success + 1))
    else
        echo "[FAIL] failure: $dataset (see log: $log_file)"
        failed=$((failed + 1))
    fi

    echo ""
done

echo "============================================================"
echo "DoNothing Baseline benchmark done"
echo "============================================================"
echo "Total datasets: $total"
echo "Success: $success"
echo "Failure: $failed"
echo "End time: $(date)"
echo "Results directory: results/donothing/"
echo "Logs directory: logs/donothing/"
echo "============================================================"
