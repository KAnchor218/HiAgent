"""
Persistence and cache backend for HiAgentEng.

Each run receives a separate LOG_DIR from the launcher. This backend creates an
additional per-task namespace inside that directory:

    LOG_DIR/
      runs/
        <task_id>/                 # for example, "tyreworld_p0"
          trajectory.jsonl         # raw action-observation pairs for each step
          subgoals/
            subgoal_000_raw.json
            subgoal_000_summary.json
            subgoal_001_raw.json
            ...
          retrieval_logs.jsonl
          prompt_logs.jsonl
          metrics.json

Task directories are isolated. Reusing the same task_id inside one LOG_DIR
clears the previous directory before recreating it.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


_TrajectoryPair = Sequence[Tuple[str, str]]


class MemoryBackend:
    """File-backed and in-memory cache for subgoal summaries.

    The same (subgoal, trajectory) summary is generated once, then served from
    memory or disk on later steps. Raw trajectories are persisted by subgoal id
    for audit and replay.
    """

    def __init__(self, log_dir: str, task_id: str):
        self.log_dir = log_dir
        self.task_id = task_id
        self.root = os.path.join(log_dir, "runs", task_id)
        if os.path.exists(self.root):
            # Avoid stale memory if a task id is reused inside the same run.
            shutil.rmtree(self.root)
        os.makedirs(os.path.join(self.root, "subgoals"), exist_ok=True)

        self._summary_cache: Dict[int, str] = {}
        self._raw_paths: Dict[int, str] = {}

    def _subgoal_raw_path(self, idx: int) -> str:
        return os.path.join(self.root, "subgoals", f"subgoal_{idx:03d}_raw.json")

    def _subgoal_summary_path(self, idx: int) -> str:
        return os.path.join(self.root, "subgoals", f"subgoal_{idx:03d}_summary.json")

    def append_trajectory(self, step: int, action: Any, observation: Any) -> None:
        path = os.path.join(self.root, "trajectory.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"step": step, "action": action, "observation": observation},
                    ensure_ascii=False,
                )
                + "\n"
            )

    def write_subgoal_raw(
        self,
        idx: int,
        subgoal: Tuple[str, str],
        pairs: _TrajectoryPair,
    ) -> str:
        """Persist the raw action-observation pairs for a completed subgoal.

        The pairs match the cme_final.py `history[index+1:obs_index+1]` slice.
        Only the temporary summarizer input filters "check valid actions"; raw
        audit artifacts keep the original working memory.
        """
        path = self._subgoal_raw_path(idx)
        serialised_pairs: List[List[List[str]]] = [
            [[role, content] for role, content in chunk] for chunk in pairs
        ]
        subgoal_payload: List[str] = [subgoal[0], subgoal[1]] if isinstance(subgoal, tuple) else [str(subgoal), ""]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"idx": idx, "subgoal": subgoal_payload, "pairs": serialised_pairs},
                f,
                ensure_ascii=False,
                indent=2,
            )
        self._raw_paths[idx] = path
        return path

    def read_subgoal_raw(self, idx: int) -> Optional[dict]:
        path = self._subgoal_raw_path(idx)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_or_create_summary(
        self,
        idx: int,
        summarizer_fn: Callable[[List[_TrajectoryPair], List[Tuple[str, str]]], List[str]],
        pairs: _TrajectoryPair,
        subgoal: Tuple[str, str],
    ) -> str:
        """Return a cached summary or create and persist one.

        summarizer_fn follows TrajectorySummarizer.generate_summary:
        ([trajectory], [subgoal_tuple]) -> [summary_str]. The backend receives a
        callable instead of owning the summarizer so the agent controls
        summarizer reuse and tests can run without an LLM.
        """
        if idx in self._summary_cache:
            return self._summary_cache[idx]

        path = self._subgoal_summary_path(idx)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            summary = cached.get("summary", "")
            if summary:
                self._summary_cache[idx] = summary
                return summary

        # Real LLM call with the same arguments as cme_final.py.
        summary = summarizer_fn([pairs], [subgoal])[0]

        subgoal_payload: List[str] = [subgoal[0], subgoal[1]] if isinstance(subgoal, tuple) else [str(subgoal), ""]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"idx": idx, "subgoal": subgoal_payload, "summary": summary},
                f,
                ensure_ascii=False,
                indent=2,
            )
        self._summary_cache[idx] = summary
        return summary

    def log_retrieval(self, step: int, requested_ids: Sequence[int]) -> None:
        path = os.path.join(self.root, "retrieval_logs.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"step": step, "requested_subgoal_ids": list(requested_ids)},
                    ensure_ascii=False,
                )
                + "\n"
            )

    def log_prompt(self, step: int, prompt: str) -> None:
        path = os.path.join(self.root, "prompt_logs.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps({"step": step, "prompt": prompt}, ensure_ascii=False)
                + "\n"
            )

    def write_metrics(self, metrics: Dict[str, Any]) -> None:
        path = os.path.join(self.root, "metrics.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
