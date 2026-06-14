#!/usr/bin/env bash
# HiAgentEng evaluation launcher.
#
# Usage: bash run_hiagent_eng.sh [--background|-b] [task_name]
#
# Examples:
#   bash run_hiagent_eng.sh tyreworld
#   bash run_hiagent_eng.sh -b tyreworld
#
# Each invocation creates a timestamped LOG_DIR. HiAgentEng audit artifacts are
# stored under LOG_DIR/runs/<game_name>_p<problem_index>/.

set -euo pipefail

BACKGROUND=false
TASK=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --background|-b)
      BACKGROUND=true
      shift
      ;;
    *)
      TASK="$1"
      shift
      ;;
  esac
done

TASK="${TASK:-tyreworld}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_PATH="${PROJECT_PATH:-$SCRIPT_DIR}"
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-/mnt/data/tmp/data-gym-cache}"
export NLTK_DATA="${NLTK_DATA:-/home/cxw/nltk_data}"

MODEL="${MODEL:-xiaoai-gpt4-turbo}"
AGENT="${AGENT:-HiAgentEng}"
STEP="${STEP:-30}"
MEMORY_SIZE="${MEMORY_SIZE:-100}"

export EVALTASK="$TASK"

case "$TASK" in
  tyreworld|barman|jericho|gripper|blocksworld)
    CFG="eval_configs/hiagent_eng/${TASK}.yaml"
    if [ ! -f "$CFG" ]; then
      echo "[ERROR] HiAgentEng config does not exist: ${CFG}" >&2
      exit 1
    fi
    case "$TASK" in
      jericho) EVAL_TASK_ARG="jericho" ;;
      *)       EVAL_TASK_ARG="pddl"   ;;
    esac
    ;;
  *)
    echo "[ERROR] Unknown task: $TASK" >&2
    exit 1
    ;;
esac

RUN_NAME="${AGENT}_${MODEL}_${TASK}_step${STEP}_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="./logs/${RUN_NAME}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/terminal.log"

echo "==> cfg       : $CFG"
echo "==> task      : $TASK  (--tasks ${EVAL_TASK_ARG})"
echo "==> model     : $MODEL"
echo "==> agent     : $AGENT"
echo "==> log dir   : $LOG_DIR"
echo "==> log file  : $LOG_FILE"
echo "==> memory dir: $LOG_DIR/runs/<task_id>/"
echo "==> mode      : $([ "$BACKGROUND" = true ] && echo "background" || echo "foreground")"
echo

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hiagent

cd "$PROJECT_PATH"

if [ "$BACKGROUND" = true ]; then
    nohup python -u agentboard/eval_main.py \
        --cfg-path "$CFG" \
        --tasks "$EVAL_TASK_ARG" \
        --model "$MODEL" \
        --log_path "$LOG_DIR" \
        --max_num_steps "$STEP" \
        --memory_size "$MEMORY_SIZE" \
        --agent "$AGENT" \
        > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "==> Background process started (PID: $PID)"
    echo "==> Terminal log : $LOG_FILE"
    echo "==> Tail log     : tail -f $LOG_FILE"
    echo "==> Stop process : kill $PID"
else
    python -u agentboard/eval_main.py \
        --cfg-path "$CFG" \
        --tasks "$EVAL_TASK_ARG" \
        --model "$MODEL" \
        --log_path "$LOG_DIR" \
        --max_num_steps "$STEP" \
        --memory_size "$MEMORY_SIZE" \
        --agent "$AGENT" \
        2>&1 | tee "$LOG_FILE"
    echo
    echo "==> Done. Terminal log : $LOG_FILE"
fi
