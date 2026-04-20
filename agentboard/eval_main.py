import pdb
import sys
import os
import re
import wandb
import warnings
import yaml
import json
import time
import argparse
from dotenv import load_dotenv
from tasks import load_task
from llm import load_llm
from utils.logging.agent_logger import AgentLogger
from utils.logging.logger import SummaryLogger


logger = AgentLogger(__name__)
warnings.filterwarnings("ignore")

TASKS=["alfworld", "jericho", "pddl", "webshop", "webarena", "tool-query", "tool-operation", "babyai", "scienceworld", 'pddl-structured', 'babyai-train', 'neural', 'blockworld']


def parse_args():
    parser = argparse.ArgumentParser(description="Testing")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--tasks", required=True, type=str, nargs='+',help="specify the tasks")
    parser.add_argument("--model", required=True ,help="specify the models, available models are stated in the configuration file")
    parser.add_argument("--wandb", action="store_true", help="specify whether the wandb board is needed")
    parser.add_argument("--log_path", required=False, default='', help="specify the place to store the resuls")
    parser.add_argument("--project_name", required=False, default='', help="specify the project name for wandb")
    parser.add_argument("--baseline_dir", required=False, default='', help="specify the baseline loggings for wandb baseline comparison visualization")
    parser.add_argument("--max_num_steps", required=False, default=30, help="specify the maximum number of steps used to finish the problems")
    parser.add_argument("--gpu_num", required=False, default=0, help="specify the number of gpu cards", type=int)
    parser.add_argument("--engine", required=False, default=0, help="the path of model", type=str)
    # llm agent related
    parser.add_argument("--agent", required=False, default=None, help="specify the name of llm agent", type=str)
    parser.add_argument("--memory_size", required=False, default=None, help="specify the memory size", type=int)

    args = parser.parse_args()

    return args

def path_constructor(loader, node):
    path_matcher = re.compile(r'\$\{([^}^{]+)\}')
    ''' Extract the matched value, expand env variable, and replace the match '''
    # 识别 YAML 中 ${VAR_NAME} 格式的占位符，从系统环境变量中读取真实值并拼接后续路径
    # 例如：${PROJECT_PATH}/data/pddl/blocksworld.jsonl → /home/user/HiAgent/data/pddl/blocksworld.jsonl
    value = node.value
    match = path_matcher.match(value)
    env_var = match.group()[2:-1]
    return os.environ.get(env_var) + value[match.end():]

def load_config(cfg_path, args):
    # 步骤 A：注册自定义 YAML tag !path，使 YAML 解析器能自动识别并展开 ${...} 环境变量占位符
    # yaml.add_implicit_resolver 让解析器对匹配正则的值自动打上 !path 标签
    # yaml.add_constructor 指定打上 !path 标签后调用 path_constructor 完成实际展开
    path_matcher = re.compile(r'\$\{([^}^{]+)\}')
    yaml.add_implicit_resolver('!path', path_matcher)
    yaml.add_constructor('!path', path_constructor)

    # 步骤 B：读取并解析 YAML，拆分为 4 个独立配置块：
    #   llm_config   — LLM 后端配置（按模型名分层，如 gpt-4-turbo、claude2、deepseek-67b 等）
    #   agent_config — Agent 类型及通用参数（name、memory_size 等）
    #   env_config   — 各任务的环境配置（按任务名分层，包含数据路径、prompt 路径、动作检查方式等）
    #   run_config   — 运行参数（wandb 开关、日志路径、项目名、最大步数等）
    with open(cfg_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    llm_config = config["llm"]
    agent_config = config["agent"]
    env_config = config["env"]
    run_config = config["run"]

    # 命令行参数优先级高于 YAML 文件，若命令行传入了对应参数则覆盖 YAML 中的值
    if args.log_path != '':
        run_config["log_path"] = args.log_path
    if args.project_name != '':
        run_config["project_name"] = args.project_name
    if args.baseline_dir != '':
        run_config["baseline_dir"] = args.baseline_dir
    # if args.gpu_num > 0:
    #     llm_config['ngpu'] = args.gpu_num
    run_config["wandb"] = args.wandb
    run_config["max_num_steps"] = int(args.max_num_steps)

    return llm_config, agent_config, env_config, run_config
  
def check_log_paths_are_ready(log_dir, baseline_dir):

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
        
    if not os.path.exists(os.path.join(log_dir, "logs")):
        os.makedirs(os.path.join(log_dir, "logs"))
    
    if not os.path.exists(baseline_dir):
        os.makedirs(baseline_dir)
    
    if not os.path.exists(os.path.join(log_dir, 'all_results.txt')):
        with open(os.path.join(log_dir, 'all_results.txt'), "w") as f:
            f.write("")
            f.close()
            
    return True

def main():
    load_dotenv()  # take environment variables from .env., load openai api key, tool key, wandb key, project path...

    # assert 'PROJECT_PATH' in os.environ
    # os.chdir(os.environ['PROJECT_PATH'])
    # print(f"os cwd={os.getcwd()}")
    # breakpoint()

    args = parse_args()
    llm_config, agent_config, env_config, run_config = load_config(args.cfg_path, args) 
    # print(f"llmconfig:\n{llm_config}")
    model = args.model
    # 按模型名从 llm_config 中索引对应的子配置（YAML llm: 节下按模型名分层）
    # 若模型名包含 'temp'，视为基于已有模型微调的本地模型：取基础模型配置后用 --engine 覆盖路径
    if 'temp' not in model:  # 标准模型，直接用模型名取对应子配置
        llm_config = llm_config[args.model]
    else:
        llm_config = llm_config[model.split(':')[-1]]  # 取冒号后的基础模型名对应的配置
        print(llm_config)
        llm_config['engine'] = args.engine  # 覆盖为本地模型路径（由 --engine 参数传入）
    if args.gpu_num > 0:
        llm_config['ngpu'] = args.gpu_num


    #---------------------------------------------- load llm -----------------------------------------------------
    logger.info("Start loading language model")
    
    if args.agent is not None:
        agent_config['name'] = args.agent
    if args.memory_size is not None:
        agent_config['memory_size'] = args.memory_size

    llm = load_llm(llm_config["name"], llm_config)
    
    logger.info("Finished loading language model")
    
    
    
    #------------------------------------------------ initialize agentboard ------------------------------------
    if not run_config.get("wandb", False):  # wandb 是实验追踪工具，不启用时禁用 wandb，agentboard 仅记录本地日志而无可视化展示
        os.environ["WANDB_MODE"] = "disabled"
        logger.info("Wandb is not enabled")
        
        wandb.init(mode="disabled")
    else:
        wandb.init(
            project=run_config["project_name"],
            name="{model_name}_{engine_name}".format(model_name=llm_config["name"], engine_name=llm_config["engine"]),
            notes="An evaluation of LLM agent",
            config= {'llm_config': llm_config,
                        'agent_config': agent_config,
                        'env_config': env_config,
                        'run_config': run_config},
            
        )

    log_dir = run_config.get("log_path", None)
    
    # 基准结果目录，用于对比
    baseline_path = run_config.get('baseline_dir', 'data/baseline_results_details')
    
    # 确认目录存在且可写 （失败则直接报错退出）        
    assert check_log_paths_are_ready(log_dir, baseline_path)
    
    # agentboard is the main launcher of visualizations and metrics calculation, 
    # 负责汇总指标、生成可视化报告（如果启用wandb的话），并且将结果记录到{log_path}/all_results.txt中
    agentboard = SummaryLogger(baseline_dir=baseline_path, log_path=log_dir) 
    
    # 读取已经跑完的任务记录存入 log_history，后续会跳过这些任务，避免重复评估。每行是一个 JSON，key是 task_name。用于实现断点续传        
    log_history = dict()
    for line in open(os.path.join(log_dir, 'all_results.txt'), "r"):
        logger.info(line)
        if "_summary" not in line:  # 过滤掉汇总行
            log_history[json.loads(line.strip())["task_name"]] = json.loads(line.strip())

    task_names = args.tasks if args.tasks != ["all"] else TASKS

    logger.info("Tested tasks: " + " ".join(log_history))
    
    #------------------------------------------------- start evaluation -------------------------------------------
    for task_name in task_names:
        
        # If the results of the task is already available at {log_path}/all_results.txt, skip the evaluation of this task to avoid rerunning. 
        # If you wish to rerun a task, make sure to remove the line recording previous task results from {log_path}/all_results.txt
        
        # if task_name in log_history:
        #     logger.info(f"Task {task_name} has been evaluated, skip")
        #
        #
        #     agentboard.log_run_result(task_name, log_history[task_name]["success_rate"], log_history[task_name]["progress_rate"], log_history[task_name]["grounding_acc"],
        #                               log_history[task_name]["success_rate_hard"], log_history[task_name]["success_rate_hard"], log_history[task_name]["progress_rate_hard"],
        #                               log_history[task_name]["progress_rate_easy"])
        #
        #     continue
        
        logger.info(f"Start evaluating task {task_name}")

        # 将 agent 通用配置与任务特有配置合并，生成本任务专用的 agent 配置
        # env_config 中的三个参数会覆盖 agent_config 的默认值：
        #   check_actions    — 动作合法性检查方式（如 "check valid actions"）
        #   check_inventory  — 背包/物品检查命令（如 "inventory"）
        #   init_prompt_path — 该任务的 prompt 模板路径（JSON 文件）
        agent_task_config = agent_config.copy()
        for key in env_config[task_name]:
            if key in ["check_actions", "check_inventory", "init_prompt_path"]:
                agent_task_config[key] = env_config[task_name][key]

        # 根据任务名加载对应的任务评估类（通过注册表动态查找）
        # tool-query 和 tool-operation 名称不同但共用同一个 tool 任务类，统一映射到 'tool'
        # 其他任务（alfworld、pddl、babyai 等）直接用任务名查找对应实现类
        if 'tool' in task_name:
            task = load_task('tool', run_config, llm_config, agent_task_config, env_config[task_name], llm=llm)
        else:
            task = load_task(task_name, run_config, llm_config, agent_task_config, env_config[task_name], llm=llm)


        # 进行评估
        success_rates, progress_rates, grounding_accs, score_state_records,\
            easy_sr, hard_sr, easy_pr, hard_pr = task.evaluate()
        
        # 计算指标
        success_rate = sum(success_rates) * 1.0 / len(success_rates)
        progress_rate = sum(progress_rates) * 1.0 / len(progress_rates)
        grounding_acc = sum(grounding_accs) * 1.0 / len(grounding_accs)

        
        logger.finish(f"Task {task_name} | Success Rate: {success_rate} , Progress Rate: {progress_rate} , Easy SR: {easy_sr}."
                      f"Hard SR: {hard_sr}, Easy PR: {easy_pr}, Hard PR: {hard_pr}, Grounding Accuracy: {grounding_acc}")
        

        # agentboard.log_run_result(task_name, success_rate, progress_rate, grounding_acc, hard_sr, easy_sr, hard_pr, easy_pr)
        
            
    logger.info("Finish evaluating all tasks")
    
    
    # agentboard.log_summary()
        

if __name__ == "__main__":
    main()
