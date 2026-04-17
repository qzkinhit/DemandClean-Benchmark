#!/bin/bash
# ============================================================
# Lopster Baseline full benchmark script
# supports --dataset argument and VERSION env var
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
echo "Lopster Baseline full benchmark"
echo "Project root: $PROJECT_ROOT"
echo "Version tag: ${VERSION:-none}"
echo "Selected dataset: ${SELECTED_DATASET:-all}"
echo "Start time: $(date)"
echo "============================================================"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate multibaseline

mkdir -p logs/lopster
mkdir -p results/lopster

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

# LopsterParameter
EPOCHS=100
LATENT_DIM=120
BATCH_SIZE=256

total=0; success=0; failed=0

for dataset_config in "${ALL_DATASETS[@]}"; do
    IFS='|' read -r dataset label_column task_type mse_attrs <<< "$dataset_config"

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

    clean_path="Data/${dataset}/clean_index.csv"
    task_name="${dataset}_lopster${VERSION:+_$VERSION}"
    log_file="logs/lopster/${task_name}.log"

    [[ ! -f "$clean_path" ]] && echo "skip: $clean_path does not exist" && failed=$((failed + 1)) && continue

    cmd="python MethodsRunScript/run_lopster/run_lopster_base.py \
        --dataset $dataset --data_path Data \
        --clean_path $clean_path \
        --task_name $task_name --output_path results/lopster/ \
        --index_attribute index --label_column $label_column \
        --task_type $task_type --models $models \
        --epochs $EPOCHS --latent_dim $LATENT_DIM --batch_size $BATCH_SIZE --verbose"

    [[ -n "$mse_attrs" ]] && cmd="$cmd --mse_attributes $(echo $mse_attrs | tr ',' ' ')"

    echo "log: $log_file"
    total=$((total + 1))
    if eval "$cmd" 2>&1 | tee "$log_file"; then
        echo "✓ success: $dataset"; success=$((success + 1))
    else
        echo "✗ failure: $dataset"; failed=$((failed + 1))
    fi
done

echo "============================================================"
echo "Lopster benchmark done: total=$total success=$success failure=$failed"
echo "============================================================"
