import pdb
import os
import json
import time
import numpy as np

from llm import load_llm
from agents import load_agent
from environment import load_environment
from common.registry import registry
from utils.logging.logger import TaskLogger
from utils.logging.agent_logger import AgentLogger

from .base_task import BaseTask

logger = AgentLogger(__name__)


@registry.register_task("pddl")
class EvalPddl(BaseTask):
    
    # default problem config: 
    
    
    def __init__(self,
                 llm_name = "gpt",
                 llm_config = None,
                 agent_name = "POMDPAgent",
                 agent_config = None,
                 env_config = None,
                 max_num_steps = 20,
                 llm = None,
                 baseline_dir = None,
                 log_path = None
                ):
        
        super().__init__()

        if llm is None:
            llm = load_llm(llm_name, llm_config)
        self.llm_name = llm_name
        self.llm = llm
        init_prompt_path = agent_config.get("init_prompt_path", None)
        agent_config["init_prompt_path"] = None
        self.agent = load_agent(agent_name, agent_config, llm)
        
        self.label_path = env_config.get("label_path", None)
        
        self.env_num_per_task = env_config.get("env_num_per_task", 1)
        self.game_name = env_config.get("game_name", []) # list of game levels
        self.problem_index = env_config.get("problem_index", None)
        self.env_configs = self.get_all_environment_configs()
        self.max_num_steps = max_num_steps
        
        
        if init_prompt_path is not None:    # load from file
            self.init_prompt_game_dict = json.load(open(init_prompt_path, 'r')) 
        
        self.baseline_dir = baseline_dir
        
        self.agentboard = TaskLogger(task_name="pddl", log_path=log_path, max_num_steps=self.max_num_steps, baseline_dir=self.baseline_dir)

    def load_seq(self, path):
        all_seqs = []
        with open(path, 'r') as f:
            for line in f:
                if line.strip() == '':
                    continue
                all_seqs.append(line.strip())
        return all_seqs
    
    def load_annotation(self, path):
        all_annotations = None  
        difficulty = []
        with open(path, 'r') as f:
            for line in f:
                if line.strip() == '':
                    continue
                line = json.loads(line.strip())
                if "difficulty" in line:
                    difficulty.append(line["difficulty"])
                else:
                    raise ValueError("No difficulty in annotation file")
        return all_annotations, difficulty
    
    def get_all_environment_configs(self):
        env_configs = []
        iter_num = 0
        _, difficulties = self.load_annotation(self.label_path)
        
        for game_name in self.game_name:
            num_problems = min(self.env_num_per_task, Num_Problems[game_name])
            if self.problem_index is not None:
                for i in self.problem_index:
                    env_configs.append({
                        "game_name": game_name,
                        "problem_index": i,
                        "difficulty": difficulties[iter_num]
                    })
                    
                    iter_num += 1
            else:
                for i in range(num_problems):
                    env_configs.append({
                        "game_name": game_name,
                        "problem_index": i,
                        "difficulty": difficulties[iter_num]
                    })
                    
                    iter_num += 1
        
        return env_configs
    
    
    def evaluate_env(self, id):
        # 加载指定id任务的评测环境和评测问题以及设置agent初始状态
        env = load_environment("pddl", self.env_configs[id])
        game_name = env.game_name
        init_obs = env._get_obs()
        goal = env._get_goal()
        self.agent.reset(goal, init_obs)
        
        # trajectory无实际作用，只是用于记录并展示轨迹，作为日志使用而不作为参数使用。
        trajectory = []
        trajectory.append({"Goal":goal, "id":0})
        trajectory.append({"Observation":init_obs, "id":0})  
        
        logger.goal("Example {} | Goal: {}".format(id, self.agent.goal))
        logger.info("Step {:02} - Message: {}".format(0, init_obs))

        # 每个样本最多尝试这么多步，超出就停止并按未完成记录。人为设定
        max_steps = self.max_num_steps
        reward = 0.
        last_reward = 0.
        grounding_acc_count = 0
        score_change_record = []  
        # 统计每题LLM的token使用情况，在任务开始时先清空
        # 只有 openai_gpt.py 和 msal_gpt.py 实现了clear_usage/get_usage 方法，所以只检查gpt
        if 'gpt' in self.llm_name:
            self.llm.clear_usage()
        self.reset_llm_runtime_stats()
        start_time = time.time()
        for step_id in range(max_steps):

            success, action = self.agent.run(init_prompt_dict = self.init_prompt_game_dict[game_name])
            
            trajectory.append({"Action":action, "id":step_id})
            logger.debug(f"success, action: {success}, action: {action}")
            if not success:
                break
            logger.info("Step {:02} - Action: {}".format(step_id, action))
            state, reward, done, infos = env.step(action)
            
            trajectory.append({"Observation":state, "id":step_id})
            trajectory.append({"Progress Rate":reward, "id":step_id})
            
            if infos.get("action_is_valid", False): 
                # 计算所用步骤中，有效的动作
                grounding_acc_count += 1
            
            # 于记录 reward 首次提升的时间点，追踪任务进度的变化历史。用于分析 agent 是在哪些步骤取得了关键进展
            if reward > last_reward:
                score_change_record.append((step_id, reward))
            last_reward = reward
            
            logger.info("Step {:02} - Observation: {}".format(step_id, state))
            logger.info("Step {:02} - Progress Rate: {}\n".format(step_id, reward))
            self.agent.update(action, state)
            # example_prompt 的作用是记录并保存 agent 每一步与 LLM 交互时实际发送的 prompt 内容
            try: example_prompt = self.agent.get_example_prompt()
            except: example_prompt = None
            trajectory.append({"Prompt": example_prompt, "id": step_id})  
            if done:
                elapsed_time = self.get_effective_elapsed_time(start_time)
                retry_overhead_time = self.get_llm_retry_overhead_time()
                env_details = {"task_name": env.game_name, "goal": self.agent.goal, "difficulty": env.difficulty,
                               "elapsed_time": round(elapsed_time, 2), "steps": step_id + 1,
                               "llm_retry_overhead_time": round(retry_overhead_time, 2)}
                # 统计token的花费情况，只有gpt模型才统计token使用情况
                if 'gpt' in self.llm_name:
                    env_details.update({'usage': self.llm.get_usage()})
                progress_rate = reward
                try: example_prompt = self.agent.get_example_prompt()
                except: example_prompt = None
                self.agentboard.log_example(id, env.won, progress_rate, grounding_acc_count / (step_id + 1), score_change_record, env_details, trajectory, example_prompt)

                return env.won, progress_rate, step_id + 1, grounding_acc_count / (step_id + 1), score_change_record


        elapsed_time = self.get_effective_elapsed_time(start_time)
        retry_overhead_time = self.get_llm_retry_overhead_time()
        env_details = {"task_name": env.game_name, "goal": self.agent.goal, "difficulty": env.difficulty,
                       "elapsed_time": round(elapsed_time, 2), "steps": step_id + 1,
                       "llm_retry_overhead_time": round(retry_overhead_time, 2)}
        if 'gpt' in self.llm_name:
            env_details.update({'usage': self.llm.get_usage()})
        try: example_prompt = self.agent.get_example_prompt()
        except: example_prompt = None

        progress_rate = reward

        self.agentboard.log_example(id, False, progress_rate, grounding_acc_count / (step_id + 1), score_change_record, env_details, trajectory, example_prompt)

        return False, progress_rate, step_id + 1, grounding_acc_count / (step_id + 1), score_change_record
                
    
    
    def evaluate(self):
        # 加载任务大类
        num_envs = len(self.env_configs)
        success_rate = []
        num_steps = []
        all_progress_rates = []
        score_state_records = []
        grounding_accs = []
        difficulties = []
        
        # 对任务大类中的各个环境（子任务）进行评测，统计评测结果并记录日志
        for id in range(num_envs):
            success, progress_rate, steps, grounding_acc, score_change_record = self.evaluate_env(id)
            all_progress_rates.append(progress_rate)
            grounding_accs.append(grounding_acc)
            score_state_records.append(score_change_record)
            difficulties.append(self.env_configs[id]["difficulty"])
            num_steps.append(steps)
            
            if success:
                success_rate.append(1)
            else:
                success_rate.append(0)
            logger.finish("Example {} | Success: {} , Progress Rate: {} , Steps: {}\n".format(id, success, progress_rate, steps))

        sr = sum(success_rate) * 1.0 / len(success_rate)
        pr = sum(all_progress_rates) * 1.0 / len(all_progress_rates)
        # 计算有效步骤中，正确的动作占比，作为 grounding accuracy 的指标。这个指标可以用来评估 agent 在理解环境和执行动作方面的能力，尤其是在需要复杂推理和规划的任务中，可以反映 agent 是否能够正确地识别和执行那些关键的动作，从而更好地完成任务。
        gr = sum(grounding_accs) * 1.0 / len(grounding_accs)

        hard_sr = [sr for sr, difficulty in zip(success_rate, difficulties) if difficulty == "hard"]
        hard_sr = sum(hard_sr) / len(hard_sr) if len(hard_sr) > 0 else 0

        hard_pr = [pr for pr, difficulty in zip(all_progress_rates, difficulties) if difficulty == "hard"]
        hard_pr = sum(hard_pr) / len(hard_pr) if len(hard_pr) > 0 else 0

        easy_sr = [sr for sr, difficulty in zip(success_rate, difficulties) if difficulty == "easy"]
        easy_sr = sum(easy_sr) / len(easy_sr) if len(easy_sr) > 0 else 0

        easy_pr = [pr for pr, difficulty in zip(all_progress_rates, difficulties) if difficulty == "easy"]
        easy_pr = sum(easy_pr) / len(easy_pr) if len(easy_pr) > 0 else 0


        self.agentboard.log_summary(sr, pr, gr, score_state_records, hard_sr, hard_pr, easy_sr, easy_pr)
        
        return success_rate, all_progress_rates, grounding_accs, score_state_records, easy_sr, hard_sr, easy_pr, hard_pr
    
    @classmethod
    def from_config(cls, 
                    run_config,
                    llm_config,
                    agent_config,
                    env_config,
                    llm = None  
                    ):
        env_name = env_config.get("name", "pddl")
        assert env_name == "pddl"
        
        max_num_steps = run_config.get("max_num_steps", 20)
        baseline_dir = run_config.get("baseline_dir", "data/baseline_results")
        llm_name = llm_config.get("name", "gpt")
        agent_name = agent_config.get("name", "POMDPAgent")
        log_path = run_config.get("log_path", None)
    
        return cls(llm_name = llm_name,
                 llm_config = llm_config,
                 agent_name = agent_name,
                 agent_config = agent_config,
                 env_config = env_config,
                 max_num_steps = max_num_steps,
                 llm = llm,
                 baseline_dir = baseline_dir,
                 log_path = log_path
                )
        
        
Num_Problems = {
    "barman":20, "blockworld":10,"gripper":20, "tyreworld":10, "blocks_medium": 10
}
