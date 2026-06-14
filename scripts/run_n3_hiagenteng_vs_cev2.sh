#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-logs/N3FullClean_HiAgentEng_vs_CEV2_$(date +%Y%m%d_%H%M%S)}"
MODEL="${MODEL:-xiaoai-gpt4-turbo}"
TASK="${TASK:-tyreworld}"
STEP="${STEP:-30}"
MEMORY_SIZE="${MEMORY_SIZE:-100}"
CFG="${CFG:-eval_configs/hiagent_eng/${TASK}.yaml}"

export PROJECT_PATH="${PROJECT_PATH:-$(pwd)}"
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-/mnt/data/tmp/data-gym-cache}"
export NLTK_DATA="${NLTK_DATA:-/home/cxw/nltk_data}"
export EVALTASK="$TASK"

mkdir -p "$ROOT"
echo "root=$ROOT" | tee "$ROOT/run_order.log"
echo "model=$MODEL task=$TASK step=$STEP cfg=$CFG" | tee -a "$ROOT/run_order.log"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hiagent

run_one() {
  local agent="$1"
  local seed="$2"
  local out="$ROOT/${agent}_seed${seed}"
  mkdir -p "$out"
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] START agent=$agent seed=$seed log=$out" | tee -a "$ROOT/run_order.log"
  python -u agentboard/eval_main.py \
    --cfg-path "$CFG" \
    --tasks pddl \
    --model "$MODEL" \
    --log_path "$out" \
    --max_num_steps "$STEP" \
    --memory_size "$MEMORY_SIZE" \
    --agent "$agent" \
    > "$out/terminal.log" 2>&1
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] DONE  agent=$agent seed=$seed log=$out" | tee -a "$ROOT/run_order.log"
  if [[ -f "$out/pddl.txt" ]]; then
    sed -n '/^\[SUMMARY\]/,$p' "$out/pddl.txt" | tee -a "$ROOT/run_order.log"
  fi
}

for seed in 1 2 3; do
  run_one HiAgentEng "$seed"
  run_one ContextEfficientAgentV2 "$seed"
done

echo "[$(date +%Y-%m-%d_%H:%M:%S)] ALL_DONE root=$ROOT" | tee -a "$ROOT/run_order.log"
