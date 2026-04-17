#!/bin/bash
# ============================================================
# CtxPipe Baseline end-to-end evaluation script.
# Uses the pretrained model ctx_50000 for inference.
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
echo "CtxPipe Baseline end-to-end evaluation"
echo "Project root: $PROJECT_ROOT"
echo "Version tag: ${VERSION:-none}"
echo "Selected dataset: ${SELECTED_DATASET:-all}"
echo "Start time: $(date)"
echo "============================================================"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate ctxpipe-pt112

mkdir -p logs/ctxpipe
mkdir -p results/ctxpipe

# Dataset configuration: dataset|label|label_index|task_type|mse_attrs
# Note: label_index is the 0-based column index after the index column is included.
declare -a ALL_DATASETS=(
    "adult|income|14|classification|age,fnlwgt,capital_gain,capital_loss,hours_per_week"
    "beers|style|4|classification|ibu,abv"
    "bike|cnt|15|regression|temp,atemp,hum,windspeed,cnt"
    "breast_cancer|class|9|classification|"
    "har|gt|3|clustering|"
    "mercedes|y|1|regression|y"
    "nasa|sound_pressure_level|5|regression|frequency,angle,chord_length,velocity,thickness,sound_pressure_level"
    "smartfactory|labels|18|classification|o_w_blo_power,o_w_blo_voltage,o_w_bhl_power,o_w_bhl_voltage"
    "soilmoisture|soil_moisture|2|regression|soil_moisture,soil_temperature"
)

MODELS_CLASSIFICATION="rf lr svm knn dt gb"
MODELS_REGRESSION="rf lr ridge lasso knn gb"
MODELS_CLUSTERING="kmeans agglomerative"

MODEL_TAG="ctx_50000"

total=0; success=0; failed=0

for dataset_config in "${ALL_DATASETS[@]}"; do
    IFS='|' read -r dataset label_column label_index task_type mse_attrs <<< "$dataset_config"

    if [[ -n "$SELECTED_DATASET" && "$dataset" != "$SELECTED_DATASET" ]]; then
        continue
    fi

    echo "------------------------------------------------------------"
    echo "Dataset: $dataset | Label: $label_column | Index: $label_index | Task: $task_type"
    echo "------------------------------------------------------------"

    case $task_type in
        "classification") models=$MODELS_CLASSIFICATION ;;
        "regression") models=$MODELS_REGRESSION ;;
        "clustering") models=$MODELS_CLUSTERING ;;
        *) models=$MODELS_CLASSIFICATION ;;
    esac

    dirty_path="Data/${dataset}/dirty_index.csv"
    clean_path="Data/${dataset}/clean_index.csv"
    task_name="${dataset}_ctxpipe${VERSION:+_$VERSION}"
    log_file="logs/ctxpipe/${task_name}.log"

    [[ ! -f "$dirty_path" ]] && echo "Skipping: $dirty_path does not exist" && failed=$((failed + 1)) && continue
    [[ ! -f "$clean_path" ]] && echo "Skipping: $clean_path does not exist" && failed=$((failed + 1)) && continue

    cmd="python MethodsRunScript/run_ctxpipe/run_ctxpipe_base.py \
        --dirty_path $dirty_path --clean_path $clean_path \
        --task_name $task_name --output_path results/ctxpipe/ \
        --index_attribute index --label_index $label_index --label_column $label_column \
        --task_type $task_type --models $models \
        --model_tag $MODEL_TAG"

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
echo "CtxPipe evaluation done: total=$total success=$success failed=$failed"
echo "============================================================"
