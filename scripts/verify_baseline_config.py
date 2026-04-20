#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from pathlib import Path

import yaml


KNOWN_AGENTS = {
    "VanillaAgent",
    "ReactAgent",
    "CustomReactAgent",
    "OurAgent",
    "ContextEfficientAgentV2",
}
KNOWN_TASKS = {
    "alfworld",
    "babyai",
    "jericho",
    "pddl",
    "scienceworld",
    "tool-operation",
    "tool-query",
    "webarena",
    "webshop",
}
PDDL_NUM_PROBLEMS = {
    "barman": 20,
    "blockworld": 10,
    "blocks_medium": 10,
    "gripper": 20,
    "tyreworld": 10,
}
JERICHO_Z8_GAMES = {"afflicted", "anchor", "snacktime", "partyfoul"}
PATH_PATTERN = re.compile(r"\$\{([^}^{]+)\}")


def path_constructor(loader, node):
    value = node.value
    match = PATH_PATTERN.match(value)
    if not match:
        return value
    env_var = match.group(1)
    env_value = os.environ.get(env_var)
    if env_value is None:
        raise KeyError(f"Missing environment variable: {env_var}")
    return env_value + value[match.end() :]


yaml.add_implicit_resolver("!path", PATH_PATTERN, None)
yaml.add_constructor("!path", path_constructor)


def load_yaml(path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)


def count_nonempty_lines(path):
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_file(path, label, errors):
    if not path.is_file():
        errors.append(f"{label} not found: {path}")


def ensure_dir(path, label, errors):
    if not path.is_dir():
        errors.append(f"{label} not found: {path}")


def validate_model(config, model, errors, checks):
    llm_cfg = config.get("llm", {})
    if model not in llm_cfg:
        errors.append(f"model '{model}' is not defined in llm config")
        return
    entry = llm_cfg[model]
    required_fields = ["name", "engine", "context_length", "max_tokens"]
    missing = [field for field in required_fields if field not in entry]
    if missing:
        errors.append(f"llm.{model} is missing fields: {', '.join(missing)}")
        return
    checks.append(
        f"llm '{model}' -> engine={entry['engine']}, context_length={entry['context_length']}, max_tokens={entry['max_tokens']}"
    )


def validate_agent(config, agent_override, warnings, errors, checks):
    agent_cfg = config.get("agent", {})
    agent_name = agent_override or agent_cfg.get("name")
    if agent_name not in KNOWN_AGENTS:
        errors.append(f"agent '{agent_name}' is not a known registered agent")
    if agent_override and agent_override != agent_cfg.get("name"):
        warnings.append(
            f"agent override '{agent_override}' differs from config default '{agent_cfg.get('name')}'"
        )
    if agent_cfg.get("need_goal") is not True:
        errors.append("agent.need_goal should be True for baseline reproduction")
    checks.append(
        f"agent '{agent_name}' with memory_size={agent_cfg.get('memory_size')} and use_parser={agent_cfg.get('use_parser')}"
    )
    return agent_name


def validate_prompt_has_game(prompt_path, game_names, errors, checks):
    ensure_file(prompt_path, "prompt file", errors)
    if errors:
        return
    prompt_json = load_json(prompt_path)
    missing_games = [name for name in game_names if name not in prompt_json]
    if missing_games:
        errors.append(
            f"prompt file {prompt_path} does not contain game entries: {', '.join(missing_games)}"
        )
        return
    checks.append(f"prompt file {prompt_path.name} covers games: {', '.join(game_names)}")


def validate_pddl(env_cfg, errors, warnings, checks):
    game_names = env_cfg.get("game_name", [])
    if not game_names:
        errors.append("env.pddl.game_name is empty")
        return

    unknown_games = [name for name in game_names if name not in PDDL_NUM_PROBLEMS]
    if unknown_games:
        errors.append(f"unknown pddl games: {', '.join(unknown_games)}")
        return

    prompt_path = Path(env_cfg.get("init_prompt_path", ""))
    label_path = Path(env_cfg.get("label_path", ""))
    ensure_file(label_path, "label file", errors)
    validate_prompt_has_game(prompt_path, game_names, errors, checks)
    if errors:
        return

    problem_index = env_cfg.get("problem_index")
    env_num_per_task = int(env_cfg.get("env_num_per_task", 0))

    if problem_index is not None:
        for game_name in game_names:
            max_problem = PDDL_NUM_PROBLEMS[game_name]
            bad_indices = [idx for idx in problem_index if idx < 0 or idx >= max_problem]
            if bad_indices:
                errors.append(
                    f"problem_index contains out-of-range values for {game_name}: {bad_indices}"
                )
        expected_examples = len(problem_index) * len(game_names)
        if env_num_per_task and env_num_per_task != PDDL_NUM_PROBLEMS[game_names[0]]:
            warnings.append(
                "problem_index is set, so env_num_per_task no longer controls the actual sample count"
            )
    else:
        expected_examples = sum(
            min(env_num_per_task, PDDL_NUM_PROBLEMS[game_name]) for game_name in game_names
        )

    label_count = count_nonempty_lines(label_path)
    if label_count < expected_examples:
        errors.append(
            f"label file only has {label_count} rows, fewer than required {expected_examples}"
        )
        return

    checks.append(
        f"pddl dataset={label_path.name}, games={','.join(game_names)}, evaluated_examples={expected_examples}, label_rows={label_count}"
    )


def validate_jericho(env_cfg, errors, checks):
    prompt_path = Path(env_cfg.get("init_prompt_path", ""))
    label_path = Path(env_cfg.get("label_path", ""))
    game_dir = Path(env_cfg.get("game_dir", ""))
    game_names = env_cfg.get("game_name", [])

    if not game_names:
        errors.append("env.jericho.game_name is empty")
        return

    ensure_file(prompt_path, "prompt file", errors)
    ensure_file(label_path, "label file", errors)
    ensure_dir(game_dir, "game directory", errors)
    if errors:
        return

    prompt_json = load_json(prompt_path)
    for field in ("examples", "instruction", "system_msg"):
        if field not in prompt_json:
            errors.append(f"jericho prompt file is missing '{field}'")

    label_count = count_nonempty_lines(label_path)
    if label_count < len(game_names):
        errors.append(
            f"label file only has {label_count} rows, fewer than required {len(game_names)}"
        )

    missing_games = []
    for game_name in game_names:
        suffix = ".z8" if game_name in JERICHO_Z8_GAMES else ".z5"
        game_file = game_dir / f"{game_name}{suffix}"
        if not game_file.is_file():
            missing_games.append(game_file.name)
    if missing_games:
        errors.append(f"missing jericho game files: {', '.join(missing_games)}")
        return

    checks.append(
        f"jericho dataset={label_path.name}, games={len(game_names)}, prompt={prompt_path.name}, game_dir={game_dir}"
    )


def main():
    parser = argparse.ArgumentParser(description="Validate baseline reproduction configs.")
    parser.add_argument("--cfg-path", required=True, help="Path to yaml config.")
    parser.add_argument("--task", required=True, help="Task key passed to eval_main.py.")
    parser.add_argument("--model", required=True, help="Model key under llm: in yaml.")
    parser.add_argument("--project-path", default=None, help="Project root for ${PROJECT_PATH}.")
    parser.add_argument("--agent", default=None, help="Optional agent override.")
    args = parser.parse_args()

    cfg_path = Path(args.cfg_path).resolve()
    if args.project_path:
        os.environ["PROJECT_PATH"] = str(Path(args.project_path).resolve())
    else:
        os.environ.setdefault("PROJECT_PATH", str(cfg_path.parents[2]))

    checks = []
    warnings = []
    errors = []

    if args.task not in KNOWN_TASKS:
        errors.append(f"task '{args.task}' is not supported")
    if not cfg_path.is_file():
        errors.append(f"config file not found: {cfg_path}")
    if errors:
        for item in errors:
            print(f"[ERROR] {item}")
        return 1

    config = load_yaml(cfg_path)
    validate_model(config, args.model, errors, checks)
    validate_agent(config, args.agent, warnings, errors, checks)

    env_cfg = config.get("env", {})
    if args.task not in env_cfg:
        errors.append(f"env section '{args.task}' is missing from config")
    else:
        if args.task == "pddl":
            validate_pddl(env_cfg[args.task], errors, warnings, checks)
        elif args.task == "jericho":
            validate_jericho(env_cfg[args.task], errors, checks)

    if errors:
        for item in errors:
            print(f"[ERROR] {item}")
        for item in warnings:
            print(f"[WARN]  {item}")
        return 1

    for item in checks:
        print(f"[OK]    {item}")
    for item in warnings:
        print(f"[WARN]  {item}")
    print(f"[PASS]  {cfg_path.name} is ready for baseline reproduction.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
