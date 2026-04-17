#!/bin/bash
# ============================================================
# ActiveClean Baseline end-to-end evaluation script.
# Note: ActiveClean requires ground truth for iterative training (Type 3).
# Supports --dataset argument and the VERSION environment variable.
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

VERSION="${VERSION:-}"

# Parse command-line arguments.
SELECTED_DATASET=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) SELECTED_DATASET="$2"; shift 2 ;;
        --version) VERSION="$2"; shift 2 ;;
        *) shift ;;
    esac
done

echo "============================================================"
echo "ActiveClean Baseline end-to-end evaluation"
echo "Project root: $PROJECT_ROOT"
echo "Version tag: ${VERSION:-none}"
echo "Selected dataset: ${SELECTED_DATASET:-all}"
echo "Start time: $(date)"
echo "============================================================"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate multibaseline

mkdir -p logs/activeclean
mkdir -p results/activeclean

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
BATCH_SIZE=50
TOTAL_BUDGET=500

total=0; success=0; failed=0

for dataset_config in "${ALL_DATASETS[@]}"; do
    IFS='|' read -r dataset label_column task_type mse_attrs <<< "$dataset_config"

    if [[ -n "$SELECTED_DATASET" && "$dataset" != "$SELECTED_DATASET" ]]; then
        continue
    fi

    echo "------------------------------------------------------------"
    echo "Dataset: $dataset | Label: $label_column | Task: $task_type"
    echo "------------------------------------------------------------"

    case $task_type in
        "classification") models=$MODELS_CLASSIFICATION ;;
        "regression") models=$MODELS_REGRESSION ;;
        "clustering") models=$MODELS_CLUSTERING ;;
        *) models=$MODELS_CLASSIFICATION ;;
    esac

    dirty_path="Data/${dataset}/dirty_index.csv"
    clean_path="Data/${dataset}/clean_index.csv"
    task_name="${dataset}_activeclean${VERSION:+_$VERSION}"
    log_file="logs/activeclean/${task_name}.log"

    [[ ! -f "$dirty_path" ]] && echo "Skipping: $dirty_path does not exist" && failed=$((failed + 1)) && continue
    [[ ! -f "$clean_path" ]] && echo "Skipping: $clean_path does not exist" && failed=$((failed + 1)) && continue

    cmd="python MethodsRunScript/run_activeclean/run_activeclean_base.py \
        --dirty_path $dirty_path --clean_path $clean_path \
        --task_name $task_name --output_path results/activeclean/ \
        --index_attribute index --label_column $label_column \
        --task_type $task_type --models $models \
        --batch_size $BATCH_SIZE --total_budget $TOTAL_BUDGET --verbose"

    [[ -n "$mse_attrs" ]] && cmd="$cmd --mse_attributes $(echo $mse_attrs | tr ',' ' ')"

    echo "Log: $log_file"
    total=$((total + 1))
    if eval "$cmd" 2>&1 | tee "$log_file"; then
        echo "[OK] $dataset"; success=$((success + 1))
    else
        echo "[FAIL] $dataset"; failed=$((failed + 1))
    fi
done

echo "============================================================"
echo "ActiveClean evaluation done: total=$total success=$success failed=$failed"
echo "============================================================"
