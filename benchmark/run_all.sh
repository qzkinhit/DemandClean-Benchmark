#!/bin/bash
# ============================================================
# DemandClean-Benchmark - Run All Baselines
# ============================================================
#
# Usage:
#   bash run_all.sh --version v1                    # run all sequentially
#   bash run_all.sh --version v1 --parallel         # run all in parallel
#   bash run_all.sh --version v1 --baseline lopster # run specific baseline
#   bash run_all.sh --version v1 --dataset beers    # run specific dataset
#   bash run_all.sh --version v1 --status           # check running tasks
#   bash run_all.sh --version v1 --stop             # stop all tasks
#
# ============================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# All baselines
ALL_BASELINES=(
    "donothing" "deleteall" "repairall" "simpleimputer" "mlimputer"
    "boostclean" "activeclean" "raha_baran" "horizon" "holoclean"
    "uniclean" "lopster" "ctxpipe"
)

show_help() {
    echo "Usage: bash run_all.sh --version <name> [options]"
    echo ""
    echo "Required:"
    echo "  --version <name>     Version identifier (e.g., v1, test)"
    echo ""
    echo "Options:"
    echo "  --parallel           Run in parallel (one screen per baseline)"
    echo "  --baseline <name>    Run only the specified baseline"
    echo "  --dataset <name>     Run only the specified dataset"
    echo "  --list               List all available baselines"
    echo "  --status             Show running tasks"
    echo "  --stop               Stop all running tasks"
    echo "  --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  bash run_all.sh --version v1"
    echo "  bash run_all.sh --version v1 --parallel"
    echo "  bash run_all.sh --version v1 --baseline lopster"
    echo "  bash run_all.sh --version v1 --baseline lopster --dataset beers"
}

# Parse arguments
VERSION=""
PARALLEL=false
SELECTED_BASELINE=""
SELECTED_DATASET=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --version|-v) VERSION="$2"; shift 2 ;;
        --parallel) PARALLEL=true; shift ;;
        --baseline) SELECTED_BASELINE="$2"; shift 2 ;;
        --dataset) SELECTED_DATASET="$2"; shift 2 ;;
        --list)
            echo "Available baselines:"
            for b in "${ALL_BASELINES[@]}"; do echo "  - $b"; done
            exit 0 ;;
        --status)
            [ -z "$VERSION" ] && { echo "Error: --version required"; exit 1; }
            echo "=== Running tasks (${VERSION}) ==="
            screen -ls 2>/dev/null | grep "_${VERSION}" || echo "  None"
            exit 0 ;;
        --stop)
            [ -z "$VERSION" ] && { echo "Error: --version required"; exit 1; }
            echo "Stopping all ${VERSION} tasks..."
            for s in $(screen -ls 2>/dev/null | grep "_${VERSION}" | awk '{print $1}'); do
                screen -X -S "$s" quit && echo "  Stopped: $s"
            done
            exit 0 ;;
        --help|-h) show_help; exit 0 ;;
        *) echo "Unknown argument: $1"; show_help; exit 1 ;;
    esac
done

# Validate arguments
[ -z "$VERSION" ] && { echo "Error: --version is required"; show_help; exit 1; }

# Determine which baselines to run
if [ -n "$SELECTED_BASELINE" ]; then
    BASELINES=("$SELECTED_BASELINE")
else
    BASELINES=("${ALL_BASELINES[@]}")
fi

# Export VERSION for sub-scripts
export VERSION

echo "============================================================"
echo "DemandClean-Benchmark - Running Baselines"
echo "============================================================"
echo "Version:   $VERSION"
echo "Baselines: ${BASELINES[*]}"
echo "Dataset:   ${SELECTED_DATASET:-all}"
echo "Mode:      $([ "$PARALLEL" = true ] && echo "parallel" || echo "sequential")"
echo "============================================================"

mkdir -p logs

# Build arguments for run.sh
RUN_ARGS=""
[ -n "$SELECTED_DATASET" ] && RUN_ARGS="--dataset $SELECTED_DATASET"

for baseline in "${BASELINES[@]}"; do
    script="MethodsRunScript/run_${baseline}/run.sh"

    if [ ! -f "$script" ]; then
        echo "Skipping $baseline: script not found"
        continue
    fi

    if [ "$PARALLEL" = true ]; then
        screen_name="${baseline}_${VERSION}"
        echo "Starting $baseline (screen: $screen_name)"
        screen -dmS "$screen_name" bash -c "
            cd $PROJECT_ROOT
            export VERSION=$VERSION
            bash $script $RUN_ARGS 2>&1 | tee logs/${baseline}_${VERSION}.log
        "
    else
        echo ""
        echo ">>> Running $baseline"
        bash "$script" $RUN_ARGS 2>&1 | tee "logs/${baseline}_${VERSION}.log"
    fi
done

echo ""
echo "============================================================"
if [ "$PARALLEL" = true ]; then
    echo "All tasks started in background"
    echo "Check:  bash run_all.sh --version $VERSION --status"
    echo "Stop:   bash run_all.sh --version $VERSION --stop"
else
    echo "All tasks completed"
fi
echo "============================================================"
