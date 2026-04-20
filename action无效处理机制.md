# Action 无效处理机制

## 一、判断 action 无效的两种情况

### 1. 解析失败（`text_to_action` 返回 `None`）

LLM 输出的文本中无法找到合法的谓词或对象，无法构造出 action literal。

```python
# pddl_env.py
action_literal = self.text_to_action(action)
if action_literal is None:
    obs = "The action is not valid and therefore takes no effect. Please check valid actions."
```

### 2. 执行无效（前后 observation 无变化）

action 可以被解析，但执行后环境状态与执行前完全相同，说明前置条件不满足，动作被底层环境静默忽略。

```python
# pddl_env.py
obs_temp, reward, done, infos = self.env.step(action_literal)
if obs_temp == self.last_obs:   # 状态未变化
    obs = "The action is not valid and therefore takes no effect. Please remember to satisfy the restriction of actions. You can also check valid actions."
```

---

## 二、无效 action 的完整处理链

### 第一步：环境层（`pddl_env.py`）

两种失败路径均走相同处理逻辑：

- 调用 `update_info(action, obs)`，将**错误提示文本**追加到 `self.states`
- 设置 `action_is_valid = False`
- `self.reward` 和 `self.done` **保持不变**，任务不停止
- 返回 `self._get_obs()`，即 `states[-1]`，此时已是错误提示文本

### 第二步：任务层（`pddl.py`）

```python
state, reward, done, infos = env.step(action)
# state = 错误提示文本，reward 未变

if infos.get("action_is_valid", False):   # False → 不计入 grounding_acc
    grounding_acc_count += 1

if reward > last_reward:                  # reward 未变 → 不记录进度里程碑
    score_change_record.append(...)

self.agent.update(action, state)          # 错误文本作为 observation 写入 agent memory
# 循环继续，不 break
```

### 第三步：Agent 层（下一轮 `run()`）

Agent 的 memory 中看到：

```
Action: <无效动作>
Observation: The action is not valid and therefore takes no effect. Please check valid actions.
```

Prompt 中有明确指引（`cme_final.py:215`）：

> "If the current action fails, you need to execute `check valid actions` to get a list of valid actions and select one from the list."

---

## 三、处理结果汇总

| 方面 | 处理方式 |
|---|---|
| 任务是否停止 | **不停止**，循环继续 |
| reward / done | **保持不变** |
| 给 agent 的反馈 | 错误提示文本作为 observation 写入 memory |
| grounding_acc | **不计入** |
| score_change_record | **不记录** |
| Agent 预期行为 | 调用 `check valid actions` 获取合法动作列表后重试 |

---

## 四、`check valid actions` 的实现机制

`check valid actions` 完全由**环境层**实现，agent 只需输出包含 `"check"` 的字符串即可触发。

### 环境层处理（`pddl_env.py`）

```python
def step(self, action):
    if "check" in action.lower():
        obs = "Valid actions are: " + ", ".join(sorted(self._get_action_space()))
        self.update_info(action, obs)
        self.infos["action_is_valid"] = True   # 计入 grounding_acc
        return self._get_obs(), self.reward, self.done, self.infos
```

`_get_action_space()` 基于**当前状态**动态枚举所有合法动作：

```python
def _get_action_space(self):
    return [self.literal_to_text(literal)
            for literal in self.env.action_space.all_ground_literals(self.last_obs)
           ] + ["check valid actions"]
```

`all_ground_literals(self.last_obs)` 是 pddlgym 底层接口，根据当前状态实时计算满足前置条件的所有动作 literal，再通过 `literal_to_text` 转成自然语言。

### 完整交互流程

```
[Agent]  Action: check valid actions
    ↓
[Env]    检测到 "check" → 枚举当前所有合法动作
         Observation: "Valid actions are: pickup a, stack a b, unstack c d, ..."
         action_is_valid = True（计入 grounding_acc）
    ↓
[Agent]  memory 中看到合法动作列表 → 从中选一个执行
         Action: pickup a
```

### 关键设计点

| 问题 | 设计 |
|---|---|
| agent 如何触发？ | 输出含 `"check"` 的字符串即可，无需精确匹配 |
| 动作列表是静态的吗？ | **不是**，每次调用都基于 `self.last_obs` 实时计算，反映当前状态 |
| 会消耗步数吗？ | 会（`update_info` 里 `self.steps += 1`），但不改变 reward / done |
| grounding_acc 计入吗？ | **计入**（`action_is_valid = True`），与真实动作同等对待 |
