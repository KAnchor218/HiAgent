#!/usr/bin/env python3
"""Lightweight regression test for HiAgentEng's summary cache backend.

This test does not call any LLM API. It verifies that MemoryBackend:
- calls the summarizer exactly once for a new subgoal summary;
- serves later requests from memory cache;
- serves a new backend instance from the on-disk summary cache.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agentboard"))

from agents.memory_backend import MemoryBackend


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="hiagent_eng_cache_"))
    calls = {"n": 0}

    def fake_summarizer(trajectories, subgoals):
        calls["n"] += 1
        return [f"summary:{subgoals[0][1]}:{len(trajectories[0])}"]

    try:
        backend = MemoryBackend(str(tmp), "tyreworld_p2")
        pairs = [[("Action", "open boot"), ("Observation", "Boot is open.")]]
        subgoal = ("Subgoal", "Open the boot")

        first = backend.get_or_create_summary(0, fake_summarizer, pairs, subgoal)
        second = backend.get_or_create_summary(0, fake_summarizer, pairs, subgoal)

        assert first == second
        assert calls["n"] == 1, f"expected one summarizer call, got {calls['n']}"

        # Build a backend object without clearing the existing directory so the
        # disk-cache read path can be tested directly.
        backend2 = object.__new__(MemoryBackend)
        backend2.log_dir = str(tmp)
        backend2.task_id = "tyreworld_p2"
        backend2.root = str(tmp / "runs" / "tyreworld_p2")
        backend2._summary_cache = {}
        backend2._raw_paths = {}

        third = backend2.get_or_create_summary(0, fake_summarizer, pairs, subgoal)
        assert third == first
        assert calls["n"] == 1, "disk cache should avoid another summarizer call"

        print("HiAgentEng summary cache test passed.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
