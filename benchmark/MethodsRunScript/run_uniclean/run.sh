#!/bin/bash
# ============================================================
# UniClean Baseline 完整测评脚本
# 前置要求：
# - 安装JDK 8或11
# - 设置JAVA_HOME环境变量
# - 安装PySpark
# 支持 --dataset 参数和 VERSION 环境变量
# ============================================================

set -e

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
echo "UniClean Baseline 完整测评"
echo "项目根目录: $PROJECT_ROOT"
echo "版本标识: ${VERSION:-无}"
echo "指定数据集: ${SELECTED_DATASET:-全部}"
echo "开始时间: $(date)"
echo "============================================================"

# 检查Java环境
if [ -z "$JAVA_HOME" ]; then
    echo "警告: JAVA_HOME未设置，UniClean需要Java环境"
fi

source /home/qianzekai/miniconda3/etc/profile.d/conda.sh
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

# UniClean参数
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
    task_name="${dataset}_uniclean${VERSION:+_$VERSION}"
    log_file="logs/uniclean/${task_name}.log"

    [[ ! -f "$dirty_path" ]] && echo "跳过: $dirty_path 不存在" && failed=$((failed + 1)) && continue
    [[ ! -f "$clean_path" ]] && echo "跳过: $clean_path 不存在" && failed=$((failed + 1)) && continue

    cmd="python MethodsRunScript/run_uniclean/run_uniclean_base.py \
        --dirty_path $dirty_path --clean_path $clean_path \
        --dataset $dataset \
        --task_name $task_name --output_path results/uniclean/ \
        --index_attribute index --label_column $label_column \
        --task_type $task_type --models $models \
        --single_max $SINGLE_MAX \
        --executor_memory $EXECUTOR_MEMORY --driver_memory $DRIVER_MEMORY --verbose"

    echo "日志: $log_file"
    total=$((total + 1))
    if eval "$cmd" 2>&1 | tee "$log_file"; then
        echo "✓ 成功: $dataset"; success=$((success + 1))
    else
        echo "✗ 失败: $dataset"; failed=$((failed + 1))
    fi
done

echo "============================================================"
echo "UniClean 测评完成: 总计=$total 成功=$success 失败=$failed"
echo "============================================================"
