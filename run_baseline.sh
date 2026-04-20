#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash run_baseline.sh [task] [--dry-run] [--no-conda] [--skip-validate]

Supported tasks:
  blocksworld | gripper | barman | tyreworld | jericho

Environment overrides:
  MODEL        default: xiaoai-gpt4-turbo
  AGENT        default: VanillaAgent
  STEP         default: 50
  MEMORY_SIZE  default: 100
  CONDA_ENV    default: hiagent
  LOG_ROOT     default: $PROJECT_PATH/logs
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_PATH="${PROJECT_PATH:-$ROOT_DIR}"
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-/mnt/data/tmp/data-gym-cache}"
export NLTK_DATA="${NLTK_DATA:-$HOME/nltk_data}"

MODEL="${MODEL:-xiaoai-gpt4-turbo}"
AGENT="${AGENT:-VanillaAgent}"
STEP="${STEP:-50}"
MEMORY_SIZE="${MEMORY_SIZE:-100}"
CONDA_ENV="${CONDA_ENV:-hiagent}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_PATH/logs}"

TASK="blocksworld"
TASK_SET=0
DRY_RUN=0
NO_CONDA=0
SKIP_VALIDATE=0

while (($#)); do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --dry-run)
      DRY_RUN=1
      ;;
    --no-conda)
      NO_CONDA=1
      ;;
    --skip-validate)
      SKIP_VALIDATE=1
      ;;
    -*)
      echo "[ERROR] Unknown option: $1" >&2
      usage
      exit 1
      ;;
    *)
      if [ "$TASK_SET" -eq 1 ]; then
        echo "[ERROR] Only one task can be provided." >&2
        usage
        exit 1
      fi
      TASK="$1"
      TASK_SET=1
      ;;
  esac
  shift
done

case "$TASK" in
  blocksworld|gripper|barman|tyreworld)
    CFG="$PROJECT_PATH/eval_configs/baseline/${TASK}.yaml"
    EVAL_TASK_ARG="pddl"
    ;;
  jericho)
    CFG="$PROJECT_PATH/eval_configs/baseline/jericho.yaml"
    EVAL_TASK_ARG="jericho"
    JERICHO_GAME_DIR="$PROJECT_PATH/data/jericho/z-machine-games-master/jericho-game-suite"
    if [ ! -d "$JERICHO_GAME_DIR" ]; then
      echo "[ERROR] Jericho game directory not found: $JERICHO_GAME_DIR" >&2
      exit 1
    fi
    ;;
  *)
    echo "[ERROR] Unsupported task: $TASK" >&2
    usage
    exit 1
    ;;
esac

if [ "$NO_CONDA" -eq 0 ]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda is not available. Re-run with --no-conda if your current Python env is already correct." >&2
    exit 1
  fi
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

if [ "$SKIP_VALIDATE" -eq 0 ]; then
  python "$PROJECT_PATH/scripts/verify_baseline_config.py" \
    --cfg-path "$CFG" \
    --task "$EVAL_TASK_ARG" \
    --model "$MODEL" \
    --project-path "$PROJECT_PATH" \
    --agent "$AGENT"
fi

RUN_NAME="${AGENT}_${MODEL}_${TASK}_step${STEP}_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_ROOT}/${RUN_NAME}"
LOG_FILE="${LOG_DIR}/terminal.log"
mkdir -p "$LOG_DIR"

CMD=(
  python -u agentboard/eval_main.py
  --cfg-path "$CFG"
  --tasks "$EVAL_TASK_ARG"
  --model "$MODEL"
  --log_path "$LOG_DIR"
  --max_num_steps "$STEP"
  --memory_size "$MEMORY_SIZE"
  --agent "$AGENT"
)

echo "==> cfg       : $CFG"
echo "==> task      : $TASK (--tasks ${EVAL_TASK_ARG})"
echo "==> model     : $MODEL"
echo "==> agent     : $AGENT"
echo "==> log dir   : $LOG_DIR"
echo "==> tee to    : $LOG_FILE"

if [ "$DRY_RUN" -eq 1 ]; then
  printf '==> command   :'
  printf ' %q' "${CMD[@]}"
  echo
  exit 0
fi

cd "$PROJECT_PATH"
"${CMD[@]}" 2>&1 | tee "$LOG_FILE"

echo
echo "==> Done. Terminal log : $LOG_FILE"
echo "==> Structured log     : $LOG_DIR/logs/${EVAL_TASK_ARG}.jsonl"
echo "==> Summary            : $LOG_DIR/${EVAL_TASK_ARG}.txt"
