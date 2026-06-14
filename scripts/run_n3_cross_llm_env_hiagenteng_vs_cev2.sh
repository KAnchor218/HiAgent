#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-logs/N3CrossLLMEnv_HiAgentEng_vs_CEV2_$(date +%Y%m%d_%H%M%S)}"
STEP="${STEP:-30}"
MEMORY_SIZE="${MEMORY_SIZE:-100}"
SEEDS="${SEEDS:-1 2 3}"

# Default cells use configs available in this repository. Override MATRIX to
# compare other models/tasks, e.g.
#   MATRIX="barman:gpt-4-turbo:eval_configs/hiagent_eng/barman.yaml ..."
MATRIX="${MATRIX:-barman:xiaoai-gpt4-turbo:eval_configs/hiagent_eng/barman.yaml tyreworld:xiaoai-gpt4-turbo:eval_configs/hiagent_eng/tyreworld.yaml}"

export PROJECT_PATH="${PROJECT_PATH:-$(pwd)}"
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-/mnt/data/tmp/data-gym-cache}"
export NLTK_DATA="${NLTK_DATA:-/home/cxw/nltk_data}"

mkdir -p "$ROOT"
echo "root=$ROOT" | tee "$ROOT/run_order.log"
echo "step=$STEP memory_size=$MEMORY_SIZE seeds=$SEEDS" | tee -a "$ROOT/run_order.log"
echo "matrix=$MATRIX" | tee -a "$ROOT/run_order.log"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hiagent

run_one() {
  local task="$1"
  local model="$2"
  local cfg="$3"
  local agent="$4"
  local seed="$5"
  local out="$ROOT/${task}_${model}/${agent}_seed${seed}"

  mkdir -p "$out"
  export EVALTASK="$task"

  echo "[$(date +%Y-%m-%d_%H:%M:%S)] START task=$task model=$model agent=$agent seed=$seed cfg=$cfg log=$out" | tee -a "$ROOT/run_order.log"
  python -u agentboard/eval_main.py \
    --cfg-path "$cfg" \
    --tasks pddl \
    --model "$model" \
    --log_path "$out" \
    --max_num_steps "$STEP" \
    --memory_size "$MEMORY_SIZE" \
    --agent "$agent" \
    > "$out/terminal.log" 2>&1
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] DONE  task=$task model=$model agent=$agent seed=$seed log=$out" | tee -a "$ROOT/run_order.log"
  if [[ -f "$out/pddl.txt" ]]; then
    sed -n '/^\[SUMMARY\]/,$p' "$out/pddl.txt" | tee -a "$ROOT/run_order.log"
  else
    echo "[$(date +%Y-%m-%d_%H:%M:%S)] WARN missing_pddl task=$task model=$model agent=$agent seed=$seed log=$out" | tee -a "$ROOT/run_order.log"
  fi
}

for item in $MATRIX; do
  IFS=: read -r task model cfg <<< "$item"
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] CELL_START task=$task model=$model cfg=$cfg" | tee -a "$ROOT/run_order.log"
  for seed in $SEEDS; do
    run_one "$task" "$model" "$cfg" HiAgentEng "$seed"
    run_one "$task" "$model" "$cfg" ContextEfficientAgentV2 "$seed"
  done
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] CELL_DONE task=$task model=$model cfg=$cfg" | tee -a "$ROOT/run_order.log"
done

echo "[$(date +%Y-%m-%d_%H:%M:%S)] ALL_DONE root=$ROOT" | tee -a "$ROOT/run_order.log"
