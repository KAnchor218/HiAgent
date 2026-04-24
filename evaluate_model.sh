export PROJECT_PATH='[Project Path Here]'
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"
MODEL="gpt-4-turbo"
export STEP=30
AGENT="ContextEfficientAgentV2"
export EVALTASK="blocksworld"
python agentboard/eval_main.py \
    --cfg-path eval_configs/hiagent/blocksworld.yaml\
    --tasks pddl \
    --model $MODEL  \
    --log_path ./logs/hiagent/analysis/"${AGENT}_${MODEL}_${STEP}"   \
    --project_name none \
    --baseline_dir ./data/baseline_results  \
    --max_num_steps $STEP     \
    --memory_size   100  \
    --agent $AGENT