# HiAgent 完整工作流与调用链详解

## 一、全局概览

HiAgent 本质是一个**评测框架**：它让一个 LLM Agent 在各种环境（游戏、网页、规划任务等）中执行任务，并记录评测指标。核心循环可以用一句话概括：

```
Agent 看到环境状态 → 思考并输出动作 → 环境执行动作并返回新状态 → 循环直到任务完成或步数耗尽
```

## 二、启动阶段：从命令行到 Agent 实例化

### 2.1 入口文件 `eval_main.py`

用户执行命令：
```bash
python agentboard/eval_main.py \
    --cfg-path eval_configs/hiagent/blocksworld.yaml \
    --tasks pddl \
    --model gpt-4-turbo \
    --agent ContextEfficientAgentV2 \
    --memory_size 100 \
    --max_num_steps 50
```

`eval_main.py` 的 `main()` 函数执行以下步骤：

```
第1步：parse_args()
    解析命令行参数，得到 args 对象

第2步：load_config(args.cfg_path, args)
    读取 YAML 配置文件，得到四个字典：
    - llm_config    → LLM 模型参数（engine、temperature、max_tokens 等）
    - agent_config  → Agent 参数（name、memory_size 等）
    - env_config    → 环境参数（数据路径、环境配置等）
    - run_config    → 运行参数（log_path、max_num_steps 等）

第3步：覆盖配置
    agent_config['name'] = args.agent          # "ContextEfficientAgentV2"
    agent_config['memory_size'] = args.memory_size  # 100

第4步：load_llm(llm_config["name"], llm_config)
    通过 Registry 按名字查找 LLM 类，实例化得到 llm 对象
    例如：llm_config["name"] = "gpt" → 实例化 OpenAI GPT 后端

第5步：遍历每个 task_name，调用 load_task()
```

### 2.2 `load_task()` —— Task 的实例化

```python
# tasks/__init__.py
def load_task(name, run_config, llm_config, agent_config, env_config, llm=None):
    task = registry.get_task_class(name).from_config(run_config, llm_config, agent_config, env_config, llm=llm)
    return task
```

以 alfworld 为例：
1. `registry.get_task_class("alfworld")` → 返回 `Evalalfworld` 类
2. 调用 `Evalalfworld.from_config(...)` 这个类方法
3. `from_config` 内部调用 `cls(...)` 即 `Evalalfworld.__init__(...)`

### 2.3 Task 的 `__init__()` —— Agent 在这里被创建

```python
# tasks/alfworld.py
class Evalalfworld(BaseTask):
    def __init__(self, ..., agent_name, agent_config, llm, ...):
        self.agent = load_agent(agent_name, agent_config, llm)  # ← 关键
```

### 2.4 `load_agent()` —— Agent 的实例化

```python
# agents/__init__.py
def load_agent(name, config, llm_model):
    agent = registry.get_agent_class(name).from_config(llm_model, config)
    return agent
```

1. `registry.get_agent_class("ContextEfficientAgentV2")` → 返回 `ContextEfficientAgentV2` 类
2. 调用 `ContextEfficientAgentV2.from_config(llm_model, config)`

### 2.5 Agent 的 `from_config()` 和 `__init__()`

```python
# cme_final.py
@classmethod
def from_config(cls, llm_model, config):
    memory_size = config.get("memory_size", 100)
    instruction = config.get("instruction", "")
    # ... 从 config 字典中提取所有参数
    return cls(llm_model, memory_size, examples, instruction, ...)
```

`__init__` 初始化的关键属性：
| 属性 | 含义 |
|------|------|
| `self.llm_model` | LLM 后端实例，负责调用 API 生成文本 |
| `self.memory_size` | 工作记忆窗口大小（保留最近多少步） |
| `self.memory` | 交互历史列表（核心数据结构） |
| `self.goal` | 当前任务目标 |
| `self.subgoal_idx` | 需要 retrieve 展开的子目标编号列表 |
| `self.instruction` | Prompt 中的任务指令 |
| `self.examples` | Few-shot 示例 |

### 2.6 完整实例化调用链

```
eval_main.py: main()
  → load_config()                          # 读取 YAML
  → load_llm("gpt", llm_config)           # 创建 LLM
  → load_task("alfworld", ...)             # 创建 Task
      → Evalalfworld.from_config(...)
          → Evalalfworld.__init__(...)
              → load_agent("ContextEfficientAgentV2", agent_config, llm)
                  → ContextEfficientAgentV2.from_config(llm, config)
                      → ContextEfficientAgentV2.__init__(llm, memory_size=100, ...)
```

---

## 三、评测阶段：Agent 如何运作

### 3.1 `task.evaluate()` —— 评测入口

```python
# eval_main.py
success_rates, progress_rates, ... = task.evaluate()
```

以 alfworld 为例，`evaluate()` 遍历所有题目：

```python
# alfworld.py
def evaluate(self):
    self.env = load_environment('alfworld', self.env_cfg)  # 加载环境
    for id in range(self.num_exams):           # 遍历每道题
        ob, info = self.env.reset()            # 重置环境，得到初始观测
        score, is_done, ... = self.evaluate_env(ob=ob, examples=examples, index=id)
```

### 3.2 `evaluate_env()` —— 单道题的完整交互循环

这是理解 Agent 运作的**最关键函数**，逐步拆解：

#### 第1步：初始化 Agent

```python
init_ob = ob.split('\n')[0]                          # 环境初始描述
goal = ob.split('\n')[1].split("Your task is to:")[1] # 任务目标

self.agent.reset(goal=goal, init_obs=init_ob)
```

`reset()` 做了什么：
```python
def reset(self, goal, init_obs, init_act=None):
    self.goal = goal                    # 设置目标，如 "put a clean plate in the cabinet"
    self.init_obs = init_obs            # 设置初始观测
    self.memory = [[('Observation', self.init_obs)]]  # 初始化记忆，只有一条初始观测
    self.steps = 0
    self.done = False
```

此时 memory 状态：
```
memory = [
    [('Observation', 'You are in the middle of a room...')]   ← 步骤0：初始状态
]
```

#### 第2步：主循环 —— Agent 与环境交替执行

```python
for i in range(0, self.max_num_steps):  # 最多执行 max_num_steps 步

    # ===== Agent 决策 =====
    success, action = self.agent.run(init_prompt_dict=init_prompt_dict)

    # ===== 环境执行 =====
    action = self.parseAction(action)                      # 格式化动作
    observation, reward, done, info = self.env.step(action) # 环境执行动作

    # ===== 更新 Agent 记忆 =====
    self.agent.update(action=action, state=observation)

    # ===== 检查是否完成 =====
    if done:
        return 1.0, True, ...
```

这就是 Agent 的核心工作模式：**run → env.step → update → run → env.step → update → ...**

---

## 四、Agent 内部运作详解

### 4.1 `run()` —— Agent 的一次推理

```python
def run(self, init_prompt_dict=None):
    # 第1步：构建 Prompt
    input_prompt = self.make_prompt(...)

    # 第2步：调用 LLM 生成回复
    success, action = self.llm_model.generate(system_message, input_prompt)

    # 第3步：解析 LLM 的输出
    if success:
        # 情况A：输出包含 Subgoal
        if 'Subgoal' in action:
            subgoal = ...                           # 提取子目标文本
            self.subgoal_idx = []                   # 清空 retrieve 记录
            self.memory.append([("Subgoal", subgoal)])  # 子目标写入记忆
            action = 剩余部分（Subgoal之后的Action） 

        # 情况B：动作是 retrieve(id)
        if 'retrieve(' in action:
            numbers = extract_numbers(action)       # 提取要恢复的子目标编号
            self.subgoal_idx += numbers             # 记录下来
            return self.run(init_prompt_dict)       # 递归调用 run()！
            # 递归的效果：重新构建 Prompt（此时 subgoal_idx 已更新，
            # 对应子目标的详细轨迹会被保留），再次调用 LLM

        # 情况C：普通 Action
        if self.use_parser:
            action = self.action_parser_for_special_llms(action)  # 格式清洗

    return success, action
```

### 4.2 `make_prompt()` —— 构建发给 LLM 的完整 Prompt

Prompt 由以下部分拼接而成：

```
┌─────────────────────────────────────────────────┐
│ instruction（任务指令 + HiAgent 专用规则）         │
│ examples（few-shot 示例）                         │
│ goal（任务目标，如有）                             │
│ check_actions 提示（如有）                         │
│ ─────── 以上是固定部分 ────────                    │
│ serialize_history(history)  ← 核心：压缩后的历史    │
│ "\nAction: "  ← 可选的续写引导                     │
└─────────────────────────────────────────────────┘
```

关键步骤：

```python
# 第1步：截断历史（粗粒度）
history = self.memory[-self.memory_size:]

# 第2步：序列化历史（含压缩）
input_prompt = query + serialize_history(history)

# 第3步：添加续写引导
input_prompt += "\nAction: " if self.memory[-1][0][0] == 'Subgoal' else ""

# 第4步：检查 token 是否超限，超限则继续截断
while num_of_tokens > max_context_length - max_tokens:
    history = history[1:]  # 砍掉最早一步
    重新构建 input_prompt
```

### 4.3 `serialize_history()` —— 历史压缩（HiAgent 核心贡献）

这是论文的核心算法，将完整的交互历史压缩成更短的版本：

```
输入的 history（完整）:
  步骤0: [Observation: 初始状态]
  步骤1: [Subgoal: "去厨房拿盘子"]
  步骤2: [Action: "go to kitchen", Observation: "你到了厨房"]
  步骤3: [Action: "take plate",    Observation: "你拿到了盘子"]
  步骤4: [Subgoal: "清洗盘子"]
  步骤5: [Action: "go to sink",    Observation: "你到了水槽"]
  步骤6: [Action: "clean plate",   Observation: "盘子洗干净了"]
  步骤7: [Subgoal: "把盘子放到柜子里"]   ← 最后一个子目标
  步骤8: [Action: "go to cabinet",  Observation: "你到了柜子"]

压缩后的 new_history:
  步骤0: [Observation: 初始状态]              ← 原样保留
  步骤1: [1 Subgoal: "去厨房拿盘子",
           Observation: "你拿到了盘子"]        ← 压缩！只保留最终Obs或摘要
  步骤4: [2 Subgoal: "清洗盘子",
           Observation: "盘子洗干净了"]        ← 压缩！
  步骤7: [3 Subgoal: "把盘子放到柜子里"]      ← 最后子目标，完整保留 ↓
  步骤8: [Action: "go to cabinet",
           Observation: "你到了柜子"]          ← 完整保留
```

算法流程：

```
1. 找到所有 Subgoal 的位置 → subgoal_index_list
2. 如果子目标数 ≤ 1 → 不压缩，直接返回
3. 构建 new_history:
   a. 第一个子目标之前的内容 → 原样保留
   b. 遍历除最后一个之外的每个子目标:
      - 如果在 keep_subgoal_index_list（被 retrieve 过）→ 完整保留
      - 否则，压缩为 [编号子目标, 摘要/最终Observation]
        - gripper/blocksworld → 只取最终 Observation（全观测环境）
        - 其他环境 → 调用 TrajectorySummarizer 用 LLM 生成摘要
   c. 最后一个子目标及其后续 → 完整保留（加编号）
4. 调用 vanilla_serialize_history(new_history) 转成字符串
```

### 4.4 `update()` —— 记录环境反馈

```python
def update(self, action, state):
    self.steps += 1
    self.memory.append([("Action", action), ('Observation', state)])
```

这是由 Task 在 `env.step()` 之后调用的，把 Agent 执行的动作和环境返回的观测一起写入记忆。

### 4.5 `action_parser_for_special_llms()` —— 动作格式清洗

不同的 LLM 输出格式不同，有的会输出：
```
The next action to take is:
go to kitchen
```
有的会输出：
```
Action: go to kitchen
```

这个函数尝试从各种格式中提取出纯粹的动作文本 `"go to kitchen"`。

---

## 五、一次完整交互的时序图

以 alfworld 中"把干净的盘子放到柜子"为例：

```
时间线 →

eval_main.py                 Evalalfworld              ContextEfficientAgentV2         LLM              Environment
    |                            |                            |                        |                    |
    |--- task.evaluate() ------->|                            |                        |                    |
    |                            |--- env.reset() ------------------------------------------------->|
    |                            |<----------------------------------- ob="You are in a room..." ----|
    |                            |                            |                        |                    |
    |                            |--- agent.reset(goal, ob) ->|                        |                    |
    |                            |                            | memory = [[Obs: "..."]] |                    |
    |                            |                            |                        |                    |
    |                            |===== 第1步循环 ============|========================|====================|
    |                            |                            |                        |                    |
    |                            |--- agent.run() ----------->|                        |                    |
    |                            |                            |--- make_prompt() ----->|                    |
    |                            |                            |    (构建完整Prompt)      |                    |
    |                            |                            |--- llm.generate() ---->|                    |
    |                            |                            |<-- "Subgoal: 去厨房\n   |                    |
    |                            |                            |     Action: go to       |                    |
    |                            |                            |     kitchen" -----------|                    |
    |                            |                            |                        |                    |
    |                            |                            | memory.append(Subgoal)  |                    |
    |                            |                            | 解析出 action            |                    |
    |                            |<-- (True, "go to kitchen")-|                        |                    |
    |                            |                            |                        |                    |
    |                            |--- env.step("go to kitchen") ---------------------------------------->|
    |                            |<------------------------------- ("You are in kitchen", 0.2, False) ----|
    |                            |                            |                        |                    |
    |                            |--- agent.update("go to     |                        |                    |
    |                            |    kitchen", "You are...") |                        |                    |
    |                            |                            | memory.append(          |                    |
    |                            |                            |   [Action, Obs])        |                    |
    |                            |                            |                        |                    |
    |                            |===== 第2步循环 ============|========================|====================|
    |                            |                            |                        |                    |
    |                            |--- agent.run() ----------->|                        |                    |
    |                            |                            |--- make_prompt() ----->|                    |
    |                            |                            |    (此时有Subgoal，      |                    |
    |                            |                            |     子目标数=1，         |                    |
    |                            |                            |     不压缩)             |                    |
    |                            |                            |    Prompt末尾无          |                    |
    |                            |                            |    "Action:"引导         |                    |
    |                            |                            |    （因为最后一条是Obs,   |                    |
    |                            |                            |     不是Subgoal）        |                    |
    |                            |                            |--- llm.generate() ---->|                    |
    |                            |                            |<-- "Action: take plate" |                    |
    |                            |<-- (True, "take plate") ---|                        |                    |
    |                            |                            |                        |                    |
    |                            | ... env.step → update → 继续循环 ...                 |                    |
    |                            |                            |                        |                    |
    |                            |===== 第N步：LLM 生成新子目标 =========================|====================|
    |                            |                            |                        |                    |
    |                            |--- agent.run() ----------->|                        |                    |
    |                            |                            |--- make_prompt() ----->|                    |
    |                            |                            |    (此时子目标数≥2，      |                    |
    |                            |                            |     旧子目标被压缩！)     |                    |
    |                            |                            |--- llm.generate() ---->|                    |
    |                            |                            |<-- "Subgoal: 洗盘子\n   |                    |
    |                            |                            |     Action: clean..."   |                    |
    |                            |                            |                        |                    |
    |                            |                            | subgoal_idx = []        |                    |
    |                            |                            | memory.append(Subgoal)  |                    |
    |                            |<-- (True, "clean plate") --|                        |                    |
    |                            |                            |                        |                    |
    |                            |===== 特殊情况：LLM 输出 retrieve ====================|====================|
    |                            |                            |                        |                    |
    |                            |--- agent.run() ----------->|                        |                    |
    |                            |                            |--- llm.generate() ---->|                    |
    |                            |                            |<-- "retrieve(1)" ------|                    |
    |                            |                            |                        |                    |
    |                            |                            | subgoal_idx = [1]       |                    |
    |                            |                            |--- self.run() 递归! --->|                    |
    |                            |                            |    make_prompt()         |                    |
    |                            |                            |    (子目标1的详细轨迹     |                    |
    |                            |                            |     被恢复到Prompt中)     |                    |
    |                            |                            |--- llm.generate() ---->|                    |
    |                            |                            |<-- "Action: ..." ------|                    |
    |                            |<-- (True, "...") ----------|                        |                    |
    |                            |                            |                        |                    |
    |                            |===== 任务完成 =============|========================|====================|
    |                            |                            |                        |                    |
    |                            |--- env.step() → done=True ------------------------------------------->|
    |                            |<-- return success_rate, ... |                        |                    |
    |<-- 记录评测结果 -----------|                            |                        |                    |
```

---

## 六、Memory 数据结构的变化过程

用一个具体例子展示 `self.memory` 在整个任务过程中是如何增长的：

```python
# reset() 之后
memory = [
    [('Observation', 'You are in a room.')]                    # 步骤0
]

# 第1次 run()：LLM 输出 "Subgoal: go to kitchen\nAction: go to kitchen 1"
# run() 内部把 Subgoal 写入 memory
memory = [
    [('Observation', 'You are in a room.')],                   # 步骤0
    [('Subgoal', ' go to kitchen')]                            # 步骤1（run内写入）
]
# run() 返回 action = "go to kitchen 1"

# Task 调用 env.step("go to kitchen 1")，得到 obs
# Task 调用 agent.update("go to kitchen 1", "You are in kitchen.")
memory = [
    [('Observation', 'You are in a room.')],                   # 步骤0
    [('Subgoal', ' go to kitchen')],                           # 步骤1
    [('Action', 'go to kitchen 1'), ('Observation', 'You...')]  # 步骤2（update写入）
]

# 第2次 run()：LLM 输出 "Action: take plate 1"（继续当前子目标）
# run() 返回 action = "take plate 1"
# Task 调用 update()
memory = [
    [('Observation', 'You are in a room.')],                   # 步骤0
    [('Subgoal', ' go to kitchen')],                           # 步骤1
    [('Action', 'go to kitchen 1'), ('Observation', 'You...')], # 步骤2
    [('Action', 'take plate 1'), ('Observation', 'You pick...')] # 步骤3
]

# 第3次 run()：LLM 输出 "Subgoal: clean plate\nAction: go to sinkbasin 1"
# run() 内部：先写入新 Subgoal，然后返回 action
memory = [
    [('Observation', 'You are in a room.')],                   # 步骤0
    [('Subgoal', ' go to kitchen')],                           # 步骤1
    [('Action', 'go to kitchen 1'), ('Observation', 'You...')], # 步骤2
    [('Action', 'take plate 1'), ('Observation', 'You pick...')], # 步骤3
    [('Subgoal', ' clean plate')]                              # 步骤4（run内写入）
]
# run() 返回 "go to sinkbasin 1"
# Task 调用 update()
memory = [
    ...,
    [('Subgoal', ' clean plate')],                             # 步骤4
    [('Action', 'go to sinkbasin 1'), ('Observation', '...')] # 步骤5
]
```

**此时调用 make_prompt() 时，子目标数=2，触发压缩：**
- 子目标1（"go to kitchen"，步骤1-3）→ 被压缩为 [1 Subgoal, 摘要/最终Obs]
- 子目标2（"clean plate"，步骤4-5）→ 最后一个子目标，完整保留

---

## 七、关键方法调用总结

| 方法 | 谁调用它 | 什么时候 | 做什么 |
|------|---------|---------|--------|
| `__init__()` | `from_config()` | 启动时 | 初始化 Agent 属性 |
| `from_config()` | `load_agent()` | 启动时 | 从配置字典创建 Agent 实例 |
| `reset()` | Task 的 `evaluate_env()` | 每道题开始时 | 设置目标、初始化记忆 |
| `run()` | Task 的 `evaluate_env()` 循环 | 每一步 | 构建 Prompt → 调 LLM → 解析输出 |
| `make_prompt()` | `run()` 内部 | 每次 `run()` | 拼接指令+示例+压缩后历史 |
| `serialize_history()` | `make_prompt()` 内部 | 每次构建 Prompt | 对历史进行子目标级压缩 |
| `vanilla_serialize_history()` | `serialize_history()` | 压缩完成后 | 列表结构 → 字符串 |
| `update()` | Task 的 `evaluate_env()` 循环 | `env.step()` 之后 | 写入 [Action, Observation] |
| `action_parser_for_special_llms()` | `run()` 内部 | LLM 返回后 | 清洗动作格式 |
| `get_example_prompt()` | Task 的 `evaluate_env()` | 任务结束时 | 获取最后的 Prompt（用于日志） |

---

## 八、LLM 输出的三种情况及处理

LLM 每次生成的文本会被 `run()` 方法解析为以下三种情况之一：

### 情况A：包含 Subgoal + Action
```
LLM 输出: "Subgoal: clean the plate\nAction: go to sinkbasin 1"
```
处理：
1. 提取 Subgoal 文本，写入 memory
2. 清空 `subgoal_idx`
3. 提取 Action 部分，经过 parser 清洗后返回

### 情况B：纯 Action
```
LLM 输出: "Action: take plate 1"
```
处理：
1. 经过 parser 清洗后直接返回

### 情况C：retrieve(id)
```
LLM 输出: "Action: retrieve(1, 2)"
```
处理：
1. 提取编号 [1, 2]，追加到 `subgoal_idx`
2. **递归调用 `self.run()`**
3. 递归中 `make_prompt()` 会检测到 `subgoal_idx` 非空，对应子目标不被压缩
4. LLM 看到完整轨迹后，重新生成 Action

---

## 九、总结：Agent 的角色定位

Agent 在这个系统中的角色是**决策者**，它本身不执行任何环境操作，只负责：

1. **观察**：通过 memory 中的 Observation 了解环境当前状态
2. **思考**：将历史记忆组装成 Prompt，让 LLM 推理下一步
3. **规划**：通过 Subgoal 机制将复杂任务分解为子目标
4. **记忆管理**：通过压缩旧子目标的轨迹来节省上下文长度
5. **输出动作**：返回一个 Action 字符串给 Task，由 Task 交给 Environment 执行

Agent **不知道**环境的内部实现，它只通过 Observation 文本了解世界。这就是为什么它需要 LLM 来"理解"文本描述并做出决策。
