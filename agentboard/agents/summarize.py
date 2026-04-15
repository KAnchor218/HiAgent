class TrajectorySummarizer:
    def __init__(self, llm_model):
        self.llm_model = llm_model

    def generate_summary(self, trajectories, subgoals):
        """
        Summarize trajectories for completed subgoals to save context length.

        Args:
            trajectories: list of trajectories, where each trajectory is a list of
                          action-observation pairs, e.g. [[("Action", "go east"), ("Observation", "You see...")], ...]
            subgoals: list of subgoal tuples, e.g. [("Subgoal", "Navigate to the kitchen")]

        Returns:
            list of summary strings, one per trajectory
        """
        summaries = []
        for trajectory, subgoal in zip(trajectories, subgoals):
            if not trajectory:
                summaries.append("Subgoal completed.")
                continue
            prompt = self._build_prompt(trajectory, subgoal)
            success, summary = self.llm_model.generate(
                "You are a helpful assistant that summarizes agent trajectories concisely.",
                prompt,
            )
            if success:
                summaries.append(summary.strip())
            else:
                summaries.append(self._fallback_summary(trajectory))
        return summaries

    def _build_prompt(self, trajectory, subgoal):
        subgoal_text = subgoal[1] if isinstance(subgoal, tuple) else str(subgoal)
        lines = [f"Subgoal: {subgoal_text}", "Trajectory:"]
        for pair in trajectory:
            for key, value in pair:
                lines.append(f"  {key}: {value}")
        lines.append(
            "\nPlease provide a concise summary (1-2 sentences) of what happened "
            "during this trajectory and the outcome. Focus on the key actions taken "
            "and the final result."
        )
        return "\n".join(lines)

    @staticmethod
    def _fallback_summary(trajectory):
        """Return the last observation as a fallback when LLM summarization fails."""
        for pair in reversed(trajectory):
            for key, value in pair:
                if key == "Observation":
                    return value
        return "Subgoal completed."
