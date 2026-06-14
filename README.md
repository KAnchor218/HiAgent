<div align="center">
<h1> HiAgent: Hierarchical Working Memory Management for Solving Long-Horizon Agent Tasks with Large Language Model
 </h1>
</div>

## 📖 Overview

HiAgent is a novel hierarchical working memory management framework for solving long-horizon agent tasks with large language models (LLMs). It introduces a hierarchical memory structure that helps LLMs better organize and utilize information during complex task solving.

Key features:
- 🧠 Hierarchical memory management with working memory and long-term memory
- 🎯 Effective for long-horizon tasks requiring multi-step planning
- 🔄 Dynamic memory updating and pruning mechanisms
- 📝 Structured memory format for better information organization
- 🤖 Compatible with various LLM backends
- ⚙️ HiAgentEng engineering backend with subgoal summary cache and audit logs

<div align="center">
<img src="assets/main.png" alt="HiAgent Overview">
</div>

## 🤝 Special Thanks

We build this repo based on [AgentBoard](https://github.com/hkust-nlp/AgentBoard/tree/main/agentboard) project. We would like to thank the authors for their excellent work.

## 🚀 Quick Start

### 🛠️ Build from source

- Clone this repo

- Create and activate virtual environment 🐍
```shell
conda create -n hiagent python=3.8.18
conda activate hiagent
```

- Setup [AgentBoard](https://github.com/hkust-nlp/AgentBoard/tree/main/agentboard) environment and data 📥

### 🔑 Setup environment

Before starting, please make sure you have configured <a href="https://developer.nvidia.com/cuda-toolkit">cuda</a>. If not, please configure it first.

If configured, you can check using the following commands:

- Check version information 📊
```shell
nvcc -V
```

### 🛠️ Additional Setup

- Download nltk library by running the following code: 📚
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

- Create and configure ```./agentboard/.env``` file, Environment Variables needed include: ⚡

```
PROJECT_PATH=
OPENAI_API_KEY=
```

### 🏃 Run script

If the configuration is correct and the code runs successfully, you should see a series of prompts in the terminal.
```shell
bash evaluate_model.sh
```

For the local reproduction scripts in this repository:

```shell
# Original HiAgent reproduction agent (ContextEfficientAgentV2)
bash run_eval.sh blocksworld
bash run_eval.sh gripper
bash run_eval.sh barman
bash run_eval.sh tyreworld

# Engineering-optimized HiAgent backend
bash run_hiagent_eng.sh tyreworld
MODEL=xiaoai-gpt4-turbo bash run_hiagent_eng.sh barman
```

`run_hiagent_eng.sh` writes the standard task summary under `logs/` and also
persists per-episode audit artifacts under:

```text
logs/HiAgentEng_<model>_<task>_step<step>_<timestamp>/
  runs/
    <task_id>/
      prompt_logs.jsonl
      trajectory.jsonl
      retrieval_logs.jsonl
      subgoals/
        subgoal_000_raw.json
        subgoal_000_summary.json
        ...
```

### ⚙️ HiAgentEng

`HiAgentEng` is an engineering-equivalent implementation of the paper method
`ContextEfficientAgentV2`. It keeps the same subgoal chunking, observation
summarization, and trajectory retrieval semantics, but avoids repeated summary
generation by caching each completed subgoal summary once per episode.

Main changes:

- in-memory and on-disk summary cache for completed subgoals;
- raw subgoal trajectory persistence for replay and auditing;
- prompt, trajectory, and retrieval logs under `LOG_DIR/runs/<task_id>/`;
- run/task-level cache isolation to avoid cross-run contamination.

The main implementation files are:

```text
agentboard/agents/hiagent_eng.py
agentboard/agents/memory_backend.py
eval_configs/hiagent_eng/*.yaml
run_hiagent_eng.sh
```

To run a 3-seed engineering-equivalence comparison against the original
`ContextEfficientAgentV2`:

```shell
python3 scripts/test_hiagent_eng_cache.py
bash scripts/run_n3_hiagenteng_vs_cev2.sh
```

To compare multiple task/model cells, override `MATRIX`:

```shell
MATRIX="barman:xiaoai-gpt4-turbo:eval_configs/hiagent_eng/barman.yaml tyreworld:xiaoai-gpt4-turbo:eval_configs/hiagent_eng/tyreworld.yaml" \
  bash scripts/run_n3_cross_llm_env_hiagenteng_vs_cev2.sh
```


### 📊 Visualize results

<div align="center">
<img src="assets/hiagent exp.png" alt="HiAgent Experiment Results">
</div>
