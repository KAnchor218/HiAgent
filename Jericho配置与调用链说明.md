# Jericho 配置来源、字段含义与调用链说明

本文整理了 HiAgent 中 Jericho 评测的几个常见疑问，并把实际调用链串起来说明：

- `run_eval.sh jericho` 没有单独的 `jericho.yaml` 时，配置到底从哪里来
- 当前论文复现里 Jericho 的 `max_num_steps` 为什么应按脚本里的 `30` 来理解
- `agentboard/environment/jericho_env.py` 中的 `game_goals`、`game_init_obs`、`solution_lengths` 分别做什么
- `data/jericho/test.jsonl` 中的 `goal`、`subgoals` 分别做什么
- Jericho 整条评测链路是如何从脚本一路走到环境、Agent 和评分逻辑的

---

## 一、结论先看

执行：

```bash
bash run_eval.sh jericho
```

时，HiAgent **不会去读取单独的 `eval_configs/hiagent/jericho.yaml`**。  
当前实现是：

- `run_eval.sh` 检测到 `TASK=jericho`
- 把 `--cfg-path` 固定设为 `eval_configs/hiagent/blocksworld.yaml`
- 把 `--tasks` 设为 `jericho`
- `eval_main.py` 读取整个 YAML 后，再从其中的 `env.jericho` 小节取出 Jericho 环境配置

因此，Jericho 在 HiAgent 中是：

- **复用 `blocksworld.yaml` 作为总配置文件**
- **真正使用其中的 `env.jericho` 配置块**

另外，当前论文复现口径下，Jericho 的有效 `max_num_steps` 应按脚本传给 `eval_main.py` 的 `--max_num_steps 30` 理解。`eval_main.py` 会用命令行参数覆盖 YAML 中的 `run.max_num_steps`，所以 YAML 里的同名字段不是最终生效值。

---

## 二、常见问题

### 1. HiAgent 中没有 `jericho.yaml`，那 `run_eval.sh jericho` 从哪拿配置

配置来源分两层：

#### 第一层：启动脚本路由

`run_eval.sh` 中对 Jericho 做了专门分支处理：

```bash
jericho)
  CFG="eval_configs/hiagent/blocksworld.yaml"
  EVAL_TASK_ARG="jericho"
```

也就是说，Jericho 的 `cfg-path` 实际上传给的是：

```bash
eval_configs/hiagent/blocksworld.yaml
```

#### 第二层：YAML 内部再按任务名取子配置

`agentboard/eval_main.py` 会把 YAML 解析成四部分：

- `llm_config`
- `agent_config`
- `env_config`
- `run_config`

然后在实际加载任务时，根据 `--tasks jericho` 取：

```python
env_config["jericho"]
```

因此，Jericho 虽然没有单独的 `jericho.yaml`，但它并不是“没有配置”，而是：

- 复用同一个总 YAML 文件
- 在该 YAML 文件的 `env.jericho` 段中保存 Jericho 专属配置

这一段通常包含：

- `game_name`
- `game_dir`
- `label_path`
- `check_actions`
- `check_inventory`
- `init_prompt_path`

---

### 2. `jericho_env.py` 中的 `game_goals` 是什么

`game_goals` 是一个**按游戏名索引的大任务说明表**。

例如：

- `905` 的目标是“先逃出房子，再开车逃离城市”
- `library` 的目标是“在图书馆找到 Graham Nelson 的书”
- `weapon` 的目标是“先校准再激活 Yi-Lono-Mordel”

它的主要作用是：

- 给环境生成 `goal`
- 再由环境把这个 `goal` 传给 Agent
- 最终作为提示词的一部分，让 Agent 明白这一局要完成什么

换句话说，`game_goals` 是**实际喂给 Agent 的任务目标文本**。

#### 使用位置

在 `Jericho.from_config()` 中：

1. 根据当前 `game_name`
2. 从 `game_goals[game_name]` 取出目标说明
3. 写入 `game_config["goal"]`
4. `reset()` 时赋值给 `self.goal`
5. `EvalJericho` 再把它传给 `agent.reset(goal, init_obs)`

如果某个 `game_name` 不在这个字典里，代码会使用兜底值：

```python
"Not written yet."
```

---

### 3. `jericho_env.py` 中的 `game_init_obs` 是什么

`game_init_obs` 是一个**按游戏名索引的开局观察文本表**。

它不是评分用的，而是给 Agent 一个清洗过、稳定的初始场景描述。

它的作用是：

- 给环境生成初始 observation
- 作为 `reset()` 后的首个状态
- 传给 Agent 作为初始上下文

#### 为什么要单独维护它

Jericho 原始游戏开局文本里往往带有：

- 格式噪声
- 多余换行
- 版权/头信息
- 运行时差异文本

当前实现没有直接把 `self.env.reset()` 返回的原始 `obs` 当作初始输入，而是改用 `game_init_obs` 里的人工整理版本。

所以可以把它理解为：

- `game_goals` 负责告诉 Agent “你要做什么”
- `game_init_obs` 负责告诉 Agent “你一开始看到了什么”

---

### 4. `jericho_env.py` 中的 `solution_lengths` 是什么

`solution_lengths` 是一个**参考解步数表**，记录某些游戏的大致标准解长度。

例如：

- `905: 22`
- `pentari: 34`
- `detective: 51`

从注释来看，它原本的意图是：

- 只保留解法不超过 60 步的游戏
- 或者给环境记录一个参考的 quest length

但在当前这份实现里，它基本**没有真正参与运行逻辑**，因为相关代码已经被注释掉了。

因此现状是：

- 不用它筛游戏
- 不用它限制步数
- 不用它算 reward
- 不用它判断 success

它更像是作者留下的参考信息或旧逻辑残留。

---

### 5. `test.jsonl` 里的 `goal` 和 `subgoals` 分别是什么

`data/jericho/test.jsonl` 每行对应一个评测样本，通常包含：

- `id`
- `goal`
- `difficulty`
- `subgoals`

这几个字段在当前实现中的作用并不相同。

#### `goal`

`test.jsonl` 里的 `goal` 在语义上是该样本的大目标描述。  
但在 **当前 Jericho 实现** 中，这个字段**并没有被实际接入运行链路**。

`EvalJericho.load_annotation()` 当前只读取：

- `subgoals`
- `difficulty`

没有把 `goal` 继续传给环境或 Agent。

所以在当前代码里：

- `test.jsonl.goal` 更像是一份标注文件中的冗余说明
- 实际喂给 Agent 的目标仍然来自 `game_goals`

#### `subgoals`

`subgoals` 才是 Jericho 当前评测的核心标注。

它们会被读取后写入环境配置中的：

```python
obs_to_reward
```

环境每走一步，都会检查当前 observation 是否匹配某个未完成的 subgoal：

- 匹配到一个，就记一次进度
- reward 变为 `已完成 subgoal 数 / subgoal 总数`
- 全部 subgoal 都匹配完时，`reward == 1`
- 此时环境把任务视为完成

因此，当前 Jericho 的评测语义可以概括为：

- `game_goals`：给 Agent 的大目标文本
- `test.jsonl.goal`：标注文件中保存的大目标文本，但当前未实际使用
- `test.jsonl.subgoals`：真正用于 progress reward 和完成判定的评分依据

---

### 6. `game_goals` 和 `test.jsonl.goal` 看起来都像目标，它们冲突吗

在当前仓库的绝大多数样本里，两者**不是冲突关系**，而是**两份内容基本一致的平行描述**。

也就是说：

- `game_goals["905"]`
- `test.jsonl` 第 0 条样本里的 `goal`

通常写的是同一件事。

所以多数情况下不会出问题，原因不是“所有任务都用同一个 `game_goals`”，而是：

- 每个游戏都会按自己的 `game_name` 取对应的 `game_goals[game_name]`
- 而当前数据集中的 `test.jsonl.goal` 恰好和它大体同步

#### 更准确地说

不是：

- 所有样本共用同一个 goal

而是：

- `905` 用 `game_goals["905"]`
- `acorncourt` 用 `game_goals["acorncourt"]`
- `library` 用 `game_goals["library"]`
- 不同游戏取不同条目

---

### 7. 什么时候会真的出问题

Jericho 当前实现有两个需要特别注意的同步风险。

#### 风险一：`test.jsonl.goal` 改了，但运行结果不变

因为当前代码没有读取 `test.jsonl.goal`，所以你如果只修改这个字段：

- Agent 实际拿到的目标文本不会变化
- 运行逻辑也不会变化

这类修改对当前实现是“看起来改了，实际上没生效”。

#### 风险二：`game_name` 列表与 `test.jsonl` 样本顺序错位

当前 `EvalJericho.get_all_environment_configs()` 不是按 `id` 或 `goal` 自动匹配数据，而是**按顺序硬对齐**：

1. 遍历 YAML 里的 `game_name` 列表
2. 第 `i` 个游戏直接拿 `test.jsonl` 第 `i` 行的 `subgoals` 和 `difficulty`

因此，如果你改了：

- `game_name` 的顺序
- `test.jsonl` 的顺序
- 或者两边增删样本但没有同步

就会出现：

- 目标游戏是 A
- 但评分标注却来自 B

这种错配会直接污染评测结果。

---

### 8. 当前仓库里是否存在真实错配

有一个已知错配点：

- YAML 里的游戏名写的是 `huntdark`
- `jericho_env.py` 中 `game_goals` / `game_init_obs` 的键写的是 `darkhunt`

结果是：

- `test.jsonl` 里该样本的 `goal` 和 `subgoals` 是存在的
- 评分仍能按 `subgoals` 正常进行
- 但环境在查 `game_goals["huntdark"]` 和 `game_init_obs["huntdark"]` 时找不到
- 因而会退回到 `"Not written yet."`

这意味着该样本当前存在：

- **评分标注可用**
- **Agent 输入的 goal/init_obs 错误**

对这一个样本来说，确实会影响表现。

---

## 三、Jericho 完整调用链

下面按实际执行顺序说明一次 `bash run_eval.sh jericho` 的调用过程。

### 第 1 步：Shell 启动脚本

用户执行：

```bash
bash run_eval.sh jericho
```

脚本内部做三件关键事：

1. 判断 `TASK=jericho`
2. 设置：
   - `CFG=eval_configs/hiagent/blocksworld.yaml`
   - `EVAL_TASK_ARG=jericho`
3. 检查游戏目录 `data/jericho/z-machine-games-master/jericho-game-suite/` 是否存在

随后真正执行：

```bash
python -u agentboard/eval_main.py \
    --cfg-path eval_configs/hiagent/blocksworld.yaml \
    --tasks jericho \
    --model ... \
    --log_path ... \
    --max_num_steps ... \
    --memory_size ... \
    --agent ...
```

### 第 2 步：`eval_main.py` 解析总配置

`eval_main.py` 调用 `load_config(args.cfg_path, args)`：

1. 读取 `blocksworld.yaml`
2. 解析出：
   - `llm_config`
   - `agent_config`
   - `env_config`
   - `run_config`
3. 再用命令行参数覆盖其中的部分值

其中也包括：

```python
run_config["max_num_steps"] = int(args.max_num_steps)
```

所以当前论文复现如果脚本传的是 `30`，Jericho 最终就是按 `30` 步运行；不要单独根据 YAML 中的 `run.max_num_steps` 判断口径。

然后根据：

```python
args.tasks == ["jericho"]
```

后续进入 Jericho 任务分支。

### 第 3 步：加载 Jericho 任务类

`eval_main.py` 在遍历任务名时，调用：

```python
load_task("jericho", run_config, llm_config, agent_task_config, env_config["jericho"], llm=llm)
```

这里真正传给 Jericho 的环境配置，就是：

```python
env_config["jericho"]
```

也就是 `blocksworld.yaml` 内部 `env.jericho` 这一段。

### 第 4 步：`EvalJericho` 组装每个样本的环境配置

`agentboard/tasks/jericho.py` 中，`EvalJericho` 初始化时会调用：

```python
self.get_all_environment_configs(env_config)
```

这个函数会：

1. 读取 `label_path = data/jericho/test.jsonl`
2. 调用 `load_annotation()`
3. 取出：
   - 每个样本的 `subgoals`
   - 每个样本的 `difficulty`
4. 遍历 YAML 中的 `game_name` 列表
5. 为每个游戏拼一个 `problem_config`

每个 `problem_config` 至少会包含：

- `game_name`
- `game_file`
- `obs_to_reward`（来自 `subgoals`）
- `difficulty`

注意：这里的 `obs_to_reward` 是 Jericho 打分的关键输入。

### 第 5 步：按样本加载环境

当开始评测某个样本时，`EvalJericho.evaluate_env(id)` 会调用：

```python
env = load_environment("jericho", self.env_configs[id])
```

然后注册表会进一步调用：

```python
Jericho.from_config(cfg)
```

### 第 6 步：`Jericho.from_config()` 补全 goal 和 init_obs

这一层会根据当前样本的 `game_name`：

1. 从 `game_goals` 中取出大目标文本
2. 从 `game_init_obs` 中取出初始观察文本
3. 写入：
   - `game_config["goal"]`
   - `game_config["init_obs"]`
4. 再实例化 `Jericho(...)`

此时一个 Jericho 环境对象已经拥有：

- `game_file`
- `game_name`
- `obs_to_reward`
- `difficulty`
- `goal`
- `init_obs`

### 第 7 步：环境 `reset()`，Agent 拿到初始输入

环境构造完成后会执行 `reset()`：

1. `self.goal = game_config["goal"]`
2. `self.init_obs = game_config["init_obs"]`
3. `self.states = [self.init_obs]`
4. `self.history = [("state", self.init_obs)]`

随后 `EvalJericho` 调用：

```python
init_obs = env._get_obs()
goal = env._get_goal()
self.agent.reset(goal, init_obs)
```

到这里，Agent 才真正拿到本局任务的：

- 大目标 `goal`
- 初始观察 `init_obs`

### 第 8 步：Agent 与环境交互

进入主循环后，每一步都是：

1. Agent 根据当前记忆和提示词生成动作
2. `env.step(action)` 执行动作
3. 环境返回：
   - 新 observation
   - 当前 reward
   - done
   - infos
4. Agent 用新的 observation 更新记忆

### 第 9 步：Jericho 如何计算 reward

Jericho 当前 reward 不是直接使用游戏原生分数，而是使用：

```python
obs_to_reward
```

也就是从 `test.jsonl.subgoals` 读进来的那组文本模式。

执行逻辑如下：

1. 每次 `step()` 后拿到新的 observation
2. 遍历尚未完成的 `obs_to_reward`
3. 如果 observation 文本匹配某个 subgoal
4. 就把它标记为已完成
5. `reward = 已完成 subgoal 数 / subgoal 总数`

因此：

- reward 表示当前完成度
- reward 从 `0` 逐步增长到 `1`
- 当 `reward == 1` 时，环境认为任务完成

### 第 10 步：记录结果

`EvalJericho` 在每步中会记录：

- 动作是否有效
- progress rate
- 轨迹
- 运行步数
- 难度
- 用时

最终输出：

- `success_rate`
- `progress_rate`
- `grounding_acc`
- 按 `easy/hard` 划分的结果统计

---

## 四、可以怎样理解这些字段

如果只记一句话，可以这样理解：

- `game_goals`：告诉 Agent “这局游戏的大任务是什么”
- `game_init_obs`：告诉 Agent “这局游戏一开始看到什么”
- `solution_lengths`：旧的参考步数表，当前几乎未使用
- `test.jsonl.goal`：标注文件中的大目标描述，当前未接入主链路
- `test.jsonl.subgoals`：真正参与 Jericho progress reward 计算的评分依据

---

## 五、建议的排查顺序

如果 Jericho 表现异常，建议按下面顺序检查：

1. `run_eval.sh` 是否真的把 `--cfg-path` 指到了 `blocksworld.yaml`
2. `blocksworld.yaml` 的 `env.jericho` 是否包含正确的 `game_name`、`game_dir`、`label_path`
3. `data/jericho/test.jsonl` 的顺序是否与 `game_name` 列表完全一致
4. `game_goals` / `game_init_obs` 是否包含当前 `game_name`
5. 是否存在拼写不一致，例如 `huntdark` / `darkhunt`
6. `subgoals` 文本是否与环境返回 observation 的表达方式足够匹配

---

## 六、当前实现的核心特点

Jericho 当前实现的关键特点不是“从一个样本文件里完整读取目标和评分”，而是：

- **目标文本来自代码里的手工字典**
- **评分标准来自 `test.jsonl` 的 `subgoals`**
- **样本对齐依赖 YAML 列表顺序**

这也意味着，Jericho 当前更像是：

- 一套“代码字典 + 标注文件 + 顺序对齐”共同驱动的评测实现

而不是完全数据驱动的实现。
