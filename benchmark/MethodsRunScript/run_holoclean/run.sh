#!/bin/bash
# ============================================================
# HoloClean Baseline full benchmark script
# Prerequisites:
# - Install a PostgreSQL database
# - Create database and user:
#   CREATE DATABASE holo;
#   CREATE USER holocleanuser WITH PASSWORD 'abcd1234';
#   GRANT ALL PRIVILEGES ON DATABASE holo TO holocleanuser;
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
echo "HoloClean Baseline full benchmark"
echo "Project root: $PROJECT_ROOT"
echo "Version tag: ${VERSION:-none}"
echo "Selected dataset: ${SELECTED_DATASET:-all}"
echo "Start time: $(date)"
echo "============================================================"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate multibaseline

mkdir -p logs/holoclean
mkdir -p results/holoclean

declare -a ALL_DATASETS=(
#    "adult|income|classification"
#    "beers|style|classification"
#    "bike|cnt|regression"
#    "breast_cancer|class|classification"
#    "har|gt|clustering"
#    "mercedes|y|regression"
#    "nasa|sound_pressure_level|regression"
    "smartfactory|labels|classification"
    "soilmoisture|soil_moisture|regression"
)

MODELS_CLASSIFICATION="rf lr svm knn dt gb"
MODELS_REGRESSION="rf lr ridge lasso knn gb"
MODELS_CLUSTERING="kmeans agglomerative"

# HoloClean parameters
DB_USER="holocleanuser"
DB_NAME="holo"
EPOCHS=10

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
    rule_path="Data/${dataset}/rules.txt"
    task_name="${dataset}_holoclean${VERSION:+_$VERSION}"
    log_file="logs/holoclean/${task_name}.log"

    [[ ! -f "$dirty_path" ]] && echo "skip: $dirty_path does not exist" && failed=$((failed + 1)) && continue
    [[ ! -f "$clean_path" ]] && echo "skip: $clean_path does not exist" && failed=$((failed + 1)) && continue

    cmd="python MethodsRunScript/run_holoclean/run_holoclean_base.py \
        --dirty_path $dirty_path --clean_path $clean_path \
        --task_name $task_name --output_path results/holoclean/ \
        --index_attribute index --label_column $label_column \
        --task_type $task_type --models $models \
        --db_user $DB_USER --db_name $DB_NAME --epochs $EPOCHS --verbose"

    # If a rules file exists, pass --rule_path
    [[ -f "$rule_path" ]] && cmd="$cmd --rule_path $rule_path"

    echo "log: $log_file"
    total=$((total + 1))
    if eval "$cmd" 2>&1 | tee "$log_file"; then
        echo "[OK] success: $dataset"; success=$((success + 1))
    else
        echo "[FAIL] failure: $dataset"; failed=$((failed + 1))
    fi
done

echo "============================================================"
echo "HoloClean benchmark done: total=$total success=$success failure=$failed"
echo "============================================================"
