#!/usr/bin/env bash
# HiAgent 评测启动脚本（含终端日志存档）
# 用法：bash run_eval.sh [--background|-b] [task_name]
#
#   参数：
#     --background, -b   在后台运行（终端关闭不中断）
#
#   PDDL 规划任务（均使用 --tasks pddl，各自有专属 yaml）：
#     blocksworld | gripper | barman | tyreworld
#
#   文字冒险游戏（需先准备 jericho 游戏文件，见下方说明）：
#     jericho
#
#   不传 task_name 则默认 blocksworld
#
# 示例：
#   bash run_eval.sh blocksworld                  # 前台运行
#   bash run_eval.sh -b blocksworld               # 后台运行
#   bash run_eval.sh --background                 # 后台运行，默认任务 blocksworld

set -euo pipefail

# ---------- 参数解析 ----------
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

TASK="${TASK:-blocksworld}"

# ---------- 基本配置 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_PATH="${PROJECT_PATH:-$SCRIPT_DIR}"
# 如果不需要代理就把下面两行注释掉
# export http_proxy="http://127.0.0.1:7890"
# export https_proxy="http://127.0.0.1:7890"

# 指定本地缓存，避免 tiktoken / NLTK 尝试联网下载
export TIKTOKEN_CACHE_DIR="/mnt/data/tmp/data-gym-cache"
export NLTK_DATA="/home/cxw/nltk_data"

MODEL="xiaoai-gpt4-turbo"
AGENT="ContextEfficientAgentV2"
STEP=30
MEMORY_SIZE=100

export EVALTASK="$TASK"

# ---------- 根据任务名自动路由 CFG 和 --tasks 参数 ----------
# PDDL 系各子任务：有专属 yaml，--tasks 统一传 pddl
# Jericho：无专属 yaml，复用 blocksworld.yaml（其中已含 jericho env 配置）
case "$TASK" in
  blocksworld|gripper|barman|tyreworld)
    CFG="eval_configs/hiagent/${TASK}.yaml"
    EVAL_TASK_ARG="pddl"
    ;;
  jericho)
    CFG="eval_configs/hiagent/blocksworld.yaml"
    EVAL_TASK_ARG="jericho"
    # jericho 需要 z-machine 游戏文件，检查是否存在
    JERICHO_GAME_DIR="${PROJECT_PATH}/data/jericho/z-machine-games-master/jericho-game-suite"
    if [ ! -d "$JERICHO_GAME_DIR" ]; then
      echo "[ERROR] Jericho 游戏文件不存在：$JERICHO_GAME_DIR"
      echo "  请先下载：pip install jericho 后运行："
      echo "    python -c \"import jericho; print(jericho.__file__)\""
      echo "  或从 https://github.com/microsoft/jericho 下载游戏包放到 data/jericho/"
      exit 1
    fi
    ;;
  *)
    echo "[ERROR] 未知任务：$TASK"
    echo "  可用任务：blocksworld | gripper | barman | tyreworld | jericho"
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
echo "==> log dir   : $LOG_DIR"
echo "==> log file  : $LOG_FILE"
echo "==> mode      : $([ "$BACKGROUND" = true ] && echo "后台 (background)" || echo "前台 (foreground)")"
echo

# ---------- 激活虚拟环境 ----------
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hiagent

# ---------- 启动评测 ----------
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
    echo "==> 后台进程已启动 (PID: $PID)"
    echo "==> Terminal log : $LOG_FILE"
    echo "==> 查看实时日志 : tail -f $LOG_FILE"
    echo "==> 停止进程     : kill $PID"
    echo
    echo "==> Structured log : $LOG_DIR/logs/${EVAL_TASK_ARG}.jsonl"
    echo "==> Summary        : $LOG_DIR/${EVAL_TASK_ARG}.txt"
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
    echo "==> Structured log     : $LOG_DIR/logs/${EVAL_TASK_ARG}.jsonl"
    echo "==> Summary            : $LOG_DIR/${EVAL_TASK_ARG}.txt"
fi
