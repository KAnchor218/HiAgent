"""
HiAgentEng is an engineering-optimized HiAgent backend.

It keeps the algorithmic behavior of ContextEfficientAgentV2 in cme_final.py
while removing avoidable runtime overhead:

1. Completed subgoal summaries are generated once and cached under
   LOG_DIR/runs/<task_id>/subgoals/subgoal_NNN_summary.json.
2. Raw action-observation pairs for completed subgoals are persisted as
   subgoal_NNN_raw.json for replay and audit. Online retrieval still uses the
   in-memory trajectory, matching the original behavior.
3. A single TrajectorySummarizer instance is reused within an episode.
4. Prompt copies are appended to prompt_logs.jsonl for equivalence auditing.

Memory is isolated per run and per task. The launcher creates a timestamped
LOG_DIR, and each episode stores artifacts under LOG_DIR/runs/<task_id>/.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional, Set

from common.registry import registry

from .cme_final import ContextEfficientAgentV2, extract_numbers
from .memory_backend import MemoryBackend
from .summarize import TrajectorySummarizer


@registry.register_agent("HiAgentEng")
class HiAgentEng(ContextEfficientAgentV2):
    """Engineering-optimized HiAgent implementation."""

    def __init__(
        self,
        llm_model,
        memory_size: int = 100,
        examples=None,
        instruction: str = "",
        init_prompt_path: Optional[str] = None,
        system_message: str = "You are a helpful assistant.",
        need_goal: bool = False,
        check_actions=None,
        check_inventory=None,
        use_parser: bool = True,
        log_path: Optional[str] = None,
    ):
        super().__init__(
            llm_model=llm_model,
            memory_size=memory_size,
            examples=examples if examples is not None else [],
            instruction=instruction,
            init_prompt_path=init_prompt_path,
            system_message=system_message,
            need_goal=need_goal,
            check_actions=check_actions,
            check_inventory=check_inventory,
            use_parser=use_parser,
        )
        self._log_path = log_path
        self._episode_counter = 0
        self.backend: Optional[MemoryBackend] = None
        self.task_id: Optional[str] = None
        self._summarizer: Optional[TrajectorySummarizer] = None
        self._raw_persisted: Set[int] = set()
        self._summary_cache_inmem = {}

    def reset(self, goal, init_obs, init_act=None, task_id: Optional[str] = None):
        super().reset(goal, init_obs, init_act)
        self._episode_counter += 1

        if task_id is None:
            # Fall back to a per-run episode id for tasks that do not pass a
            # stable external task id yet.
            evaltask = os.environ.get("EVALTASK", "unknown")
            task_id = f"{evaltask}_ep{self._episode_counter}"
        self.task_id = task_id

        self._raw_persisted = set()
        self._summary_cache_inmem = {}
        # Rebuild per-episode state to avoid sharing summaries across episodes.
        self._summarizer = None
        if self._log_path:
            self.backend = MemoryBackend(self._log_path, task_id)
        else:
            self.backend = None

    def update(self, action, state):
        super().update(action, state)
        if self.backend is not None:
            self.backend.append_trajectory(self.steps, action, state)

    def _ensure_summarizer(self) -> TrajectorySummarizer:
        if self._summarizer is None:
            self._summarizer = TrajectorySummarizer(self.llm_model)
        return self._summarizer

    def _persist_completed_subgoal_raws(self) -> None:
        """Persist raw trajectories for all completed subgoals.

        The persisted slices match self.memory exactly, including
        "check valid actions" entries. The original implementation only filters
        those entries in the temporary summarizer input, so audit artifacts must
        keep the unmodified working memory.
        """
        if self.backend is None:
            return
        subgoal_indices = [
            j for j, item in enumerate(self.memory) if item and item[0][0] == "Subgoal"
        ]
        # The last subgoal is still active, so it is not complete yet.
        for i in range(len(subgoal_indices) - 1):
            if i in self._raw_persisted:
                continue
            start = subgoal_indices[i]
            end_obs = subgoal_indices[i + 1] - 1
            subgoal_tuple = self.memory[start][0]
            trajectory = self.memory[start + 1 : end_obs + 1]
            self.backend.write_subgoal_raw(i, subgoal_tuple, trajectory)
            self._raw_persisted.add(i)

    def make_prompt(
        self,
        need_goal: bool = False,
        check_actions: str = "check valid actions",
        check_inventory: str = "inventory",
        system_message: str = "",
    ):
        # Persist completed raw subgoal trajectories before prompt rendering.
        self._persist_completed_subgoal_raws()

        # Keep this method aligned with cme_final.py:make_prompt. The only
        # semantic-preserving substitutions are marked with [ENG].
        def vanilla_serialize_history(history):
            res = []
            for item in history:
                for _ in item:
                    res.append(_[0] + ": " + _[1])
            return "\n".join(res)

        def serialize_history(history):
            self.task = os.environ.get("EVALTASK") or ""
            if any(_ in self.task for _ in ["gripper", "blocksworld"]):
                summarization = False
            else:
                summarization = True

            subgoal_index_list = []
            keep_subgoal_index_list = [_ - 1 for _ in self.subgoal_idx]
            for i in range(0, len(history)):
                item = history[i]
                if item[0][0] == "Subgoal":
                    subgoal_index_list.append(i)
            if len(subgoal_index_list) <= 1:
                return vanilla_serialize_history(history)

            final_subgoal = subgoal_index_list[-1]
            new_history = history[: subgoal_index_list[0]]
            for i in range(0, len(subgoal_index_list) - 1):
                if i in keep_subgoal_index_list:
                    new_history += history[subgoal_index_list[i] : subgoal_index_list[i + 1]]
                    continue
                index = subgoal_index_list[i]
                obs_index = subgoal_index_list[i + 1] - 1
                if not summarization:
                    subgoal = history[index][0]
                    _ = subgoal[0]
                    subgoal = (f"{i+1} {_}", subgoal[1])
                    new_history.append([subgoal, ("Observation", history[obs_index][1][1])])
                else:
                    # [ENG] Reuse one summarizer and cache each completed
                    # subgoal summary by subgoal index.
                    summarizer = self._ensure_summarizer()
                    subgoal = history[index][0]
                    trajectory = history[index + 1 : obs_index + 1]
                    trajectory = [
                        pair
                        for pair in trajectory
                        if pair[0][0] != "Action" or "check valid" not in pair[0][1]
                    ]

                    if self.backend is not None:
                        summary = self.backend.get_or_create_summary(
                            idx=i,
                            summarizer_fn=summarizer.generate_summary,
                            pairs=trajectory,
                            subgoal=subgoal,
                        )
                    else:
                        # Fall back to an in-memory cache when log_path is not
                        # configured.
                        cached = self._summary_cache_inmem.get(i)
                        if cached is None:
                            cached = summarizer.generate_summary([trajectory], [subgoal])[0]
                            self._summary_cache_inmem[i] = cached
                        summary = cached

                    _ = subgoal[0]
                    subgoal = (f"{i+1} {_}", subgoal[1])
                    new_history.append([subgoal, ("Observation", summary)])

            subgoal = history[final_subgoal][0]
            _ = subgoal[0]
            subgoal = (f"{len(subgoal_index_list)} {_}", subgoal[1])
            _ = [[subgoal]] + history[final_subgoal + 1 :]
            new_history += _
            return vanilla_serialize_history(new_history)

        # Defensive fallback for objects restored without __init__.
        if not hasattr(self, "_summary_cache_inmem"):
            self._summary_cache_inmem = {}

        query = ""
        _ = """
Note: A subgoal is a milestone goal that you need to complete in order to achieve the final goal.
When there is an unfinished subgoal, you need to ground the given subgoal to corresponding executable actions for solving the given task in the following format: \"Action: {action}\".
When there is no current subgoal or you believe the previous subgoal has been completed (based on past actions and observations), you need to output the next subgoal to be completed and its first action in the following format: \"Subgoal: {subgoal}\\nAction: {action}\".
Instructions:
1. You cannot output two subgoals consecutively.
2. Subgoal must be one line of text and does not print any newline characters.
3. Each subgoal must be followed by the execution of at least one valid action. If the current action fails, you need to execute "check valid actions" to get a list of valid actions and select one from the list.
4. **Detailed trajectory information (action-observation pair) of previously satisfied subgoals will be hidden for context efficiency. If you believe that the detailed trajectory information of a particular subgoal is crucial for the current subgoal, you can use Action: \"retrieve(subgoal_id_1, subgoal_id_2, ...)\" to obtain the detailed trajectory information.**
        """

        if _ not in self.instruction:
            self.instruction += _
        query += self.split["instruction"][0] + self.instruction + self.split["instruction"][-1]

        if isinstance(self.examples, str):
            self.examples = [self.examples]

        if len(self.examples) > 0:
            query += "\nHere are examples:\n" + self.split["example"][0]
            for example in self.examples:
                query += example + "\n"
            query += self.split["example"][-1]
        if need_goal:
            query += (
                self.split["goal"][0]
                + "You should perform actions to accomplish the goal: "
                + self.goal
                + "\n"
                + self.split["goal"][-1]
            )
        if check_actions is not None:
            query += (
                "You should use the following commands for help when your action cannot be understood: "
                + check_actions
                + "\n"
            )
        if check_inventory is not None:
            query += "You should use the following commands for help when your action cannot be understood: inventory\n"

        history = self.memory[-self.memory_size :]
        input_prompt = query + serialize_history(history)
        input_prompt += "\nAction: " if self.memory[-1][0][0] == "Subgoal" else ""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": input_prompt},
        ]
        num_of_tokens = self.llm_model.num_tokens_from_messages(messages)
        while num_of_tokens > self.max_context_length - self.llm_model.max_tokens:
            history = history[1:]
            input_prompt = query + serialize_history(history)
            input_prompt += "\nAction: " if self.memory[-1][0][0] == "Subgoal" else ""
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": input_prompt},
            ]
            num_of_tokens = self.llm_model.num_tokens_from_messages(messages)
        print(
            f"------------[Prompt Start]-----------\n{input_prompt}\n----------[Prompt END]------------"
        )
        # [ENG] Write a prompt copy for audit without changing the LLM call.
        if self.backend is not None:
            self.backend.log_prompt(self.steps, input_prompt)
        return input_prompt

    def run(self, init_prompt_dict=None):
        if init_prompt_dict is not None:
            self.init_prompt_dict = init_prompt_dict
            self.instruction = init_prompt_dict["instruction"]
            self.examples = init_prompt_dict["examples"]
        system_message = self.init_prompt_dict["system_msg"]
        input_prompt = self.make_prompt(
            need_goal=self.need_goal,
            check_actions=self.check_actions,
            check_inventory=self.check_inventory,
            system_message=system_message,
        )
        self.log_example_prompt(input_prompt)

        success, action = self.llm_model.generate(system_message, input_prompt)
        print(
            f"-------------GPT Response---------\n{action}\n---------------[END]------------"
        )
        if success:
            is_action = "Subgoal" not in action
            if not is_action:
                subgoal = action.split("\n")[0]
                subgoal = subgoal.replace("Subgoal:", "")
                self.subgoal_idx = []
                self.memory.append([("Subgoal", subgoal)])
                action = "\n".join(action.split("\n")[1:])
            if self.use_parser:
                action = self.action_parser_for_special_llms(action)
            if "retrieve(" in action.lower():
                action = action.lower()
                numbers = extract_numbers(action)
                self.subgoal_idx += numbers
                # [ENG] Log retrieval requests without changing behavior.
                if self.backend is not None:
                    self.backend.log_retrieval(self.steps, numbers)
                return self.run(init_prompt_dict)
        return success, action

    @classmethod
    def from_config(cls, llm_model, config):
        memory_size = config.get("memory_size", 100)
        instruction = config.get("instruction", "")
        examples = config.get("examples", [])
        init_prompt_path = config.get("init_prompt_path", None)
        system_message = config.get("system_message", "You are a helpful assistant.")
        check_actions = config.get("check_actions", None)
        check_inventory = config.get("check_inventory", None)
        use_parser = config.get("use_parser", True)
        need_goal = config.get("need_goal", False)
        log_path = config.get("log_path", None)
        return cls(
            llm_model=llm_model,
            memory_size=memory_size,
            examples=examples,
            instruction=instruction,
            init_prompt_path=init_prompt_path,
            system_message=system_message,
            need_goal=need_goal,
            check_actions=check_actions,
            check_inventory=check_inventory,
            use_parser=use_parser,
            log_path=log_path,
        )
