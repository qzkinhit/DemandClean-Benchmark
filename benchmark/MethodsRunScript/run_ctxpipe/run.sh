#!/bin/bash
# ============================================================
# CtxPipe Baseline 完整测评脚本
# 使用预训练模型 ctx_50000 进行推理
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
echo "CtxPipe Baseline 完整测评"
echo "项目根目录: $PROJECT_ROOT"
echo "版本标识: ${VERSION:-无}"
echo "指定数据集: ${SELECTED_DATASET:-全部}"
echo "开始时间: $(date)"
echo "============================================================"

source /home/qianzekai/miniconda3/etc/profile.d/conda.sh
conda activate ctxpipe-pt112

mkdir -p logs/ctxpipe
mkdir -p results/ctxpipe

# 数据集配置: 数据集名称|标签列|标签索引|任务类型|mse属性
# 注意: label_index是包含index列后的列位置（从0开始）
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
    echo "数据集: $dataset | 标签: $label_column | 索引: $label_index | 任务: $task_type"
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

    [[ ! -f "$dirty_path" ]] && echo "跳过: $dirty_path 不存在" && failed=$((failed + 1)) && continue
    [[ ! -f "$clean_path" ]] && echo "跳过: $clean_path 不存在" && failed=$((failed + 1)) && continue

    cmd="python MethodsRunScript/run_ctxpipe/run_ctxpipe_base.py \
        --dirty_path $dirty_path --clean_path $clean_path \
        --task_name $task_name --output_path results/ctxpipe/ \
        --index_attribute index --label_index $label_index --label_column $label_column \
        --task_type $task_type --models $models \
        --model_tag $MODEL_TAG"

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
echo "CtxPipe 测评完成: 总计=$total 成功=$success 失败=$failed"
echo "============================================================"
