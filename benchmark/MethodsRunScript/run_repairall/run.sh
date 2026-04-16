#!/bin/bash
# ============================================================
# RepairAll Baseline 完整测评脚本
# 支持 --dataset 参数和 VERSION 环境变量
# ============================================================

set -e

trap 'echo "脚本在第 $LINENO 行失败，退出码: $?"' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

VERSION="${VERSION:-}"

# 解析命令行参数
SELECTED_DATASET=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) SELECTED_DATASET="$2"; shift 2 ;;
        --version) VERSION="$2"; shift 2 ;;
        *) shift ;;
    esac
done

echo "============================================================"
echo "RepairAll Baseline 完整测评"
echo "项目根目录: $PROJECT_ROOT"
echo "版本标识: ${VERSION:-无}"
echo "指定数据集: ${SELECTED_DATASET:-全部}"
echo "开始时间: $(date)"
echo "============================================================"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate multibaseline

mkdir -p logs/repairall
mkdir -p results/repairall

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

total=0; success=0; failed=0

for dataset_config in "${ALL_DATASETS[@]}"; do
    IFS='|' read -r dataset label_column task_type mse_attrs <<< "$dataset_config"

    if [[ -n "$SELECTED_DATASET" && "$dataset" != "$SELECTED_DATASET" ]]; then
        continue
    fi

    echo "------------------------------------------------------------"
    echo "数据集: $dataset | 标签: $label_column | 任务: $task_type"
    echo "------------------------------------------------------------"

    case $task_type in
        "classification") models=$MODELS_CLASSIFICATION ;;
        "regression") models=$MODELS_REGRESSION ;;
        "clustering") models=$MODELS_CLUSTERING ;;
        *) models=$MODELS_CLASSIFICATION ;;
    esac

    dirty_path="Data/${dataset}/dirty_index.csv"
    clean_path="Data/${dataset}/clean_index.csv"
    task_name="${dataset}_repairall${VERSION:+_$VERSION}"
    log_file="logs/repairall/${task_name}.log"

    [[ ! -f "$dirty_path" ]] && echo "跳过: $dirty_path 不存在" && failed=$((failed + 1)) && continue
    [[ ! -f "$clean_path" ]] && echo "跳过: $clean_path 不存在" && failed=$((failed + 1)) && continue

    cmd="python MethodsRunScript/run_repairall/run_repairall_base.py \
        --dirty_path $dirty_path --clean_path $clean_path \
        --task_name $task_name --output_path results/repairall/ \
        --index_attribute index --label_column $label_column \
        --task_type $task_type --models $models --verbose"

    [[ -n "$mse_attrs" ]] && cmd="$cmd --mse_attributes $(echo $mse_attrs | tr ',' ' ')"

    echo "日志: $log_file"
    total=$((total + 1))
    if eval "$cmd" 2>&1 | tee "$log_file"; then
        echo "✓ 成功: $dataset"; success=$((success + 1))
    else
        echo "✗ 失败: $dataset"; failed=$((failed + 1))
    fi
done

echo "============================================================"
echo "RepairAll 测评完成: 总计=$total 成功=$success 失败=$failed"
echo "============================================================"
