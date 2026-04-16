#!/bin/bash
# ============================================================
# DoNothing Baseline 完整测评脚本
#
# 功能：一键运行所有数据集的测评
# 支持通过 --dataset 指定单个数据集
# 支持通过 VERSION 环境变量添加版本后缀
# ============================================================

set -e

trap 'echo "脚本在第 $LINENO 行失败，退出码: $?"' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# 读取版本标识（由 run_all.sh 传入或手动设置）
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
echo "DoNothing Baseline 完整测评"
echo "项目根目录: $PROJECT_ROOT"
echo "版本标识: ${VERSION:-无}"
echo "指定数据集: ${SELECTED_DATASET:-全部}"
echo "开始时间: $(date)"
echo "============================================================"

source /home/qianzekai/miniconda3/etc/profile.d/conda.sh
conda activate multibaseline

mkdir -p logs/donothing
mkdir -p results/donothing

# 数据集配置: 数据集名称|标签列|任务类型|MSE属性
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

    # 如果指定了数据集，跳过其他数据集
    if [[ -n "$SELECTED_DATASET" && "$dataset" != "$SELECTED_DATASET" ]]; then
        continue
    fi

    echo "------------------------------------------------------------"
    echo "数据集: $dataset"
    echo "标签列: $label_column"
    echo "任务类型: $task_type"
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
        echo "警告: 脏数据文件不存在: $dirty_path，跳过..."
        failed=$((failed + 1))
        continue
    fi

    if [[ ! -f "$clean_path" ]]; then
        echo "警告: 干净数据文件不存在: $clean_path，跳过..."
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

    echo "运行命令: $cmd"
    echo "日志文件: $log_file"

    total=$((total + 1))
    if eval "$cmd" 2>&1 | tee "$log_file"; then
        echo "✓ 成功: $dataset"
        success=$((success + 1))
    else
        echo "✗ 失败: $dataset (查看日志: $log_file)"
        failed=$((failed + 1))
    fi

    echo ""
done

echo "============================================================"
echo "DoNothing Baseline 测评完成"
echo "============================================================"
echo "总数据集: $total"
echo "成功: $success"
echo "失败: $failed"
echo "结束时间: $(date)"
echo "结果目录: results/donothing/"
echo "日志目录: logs/donothing/"
echo "============================================================"
