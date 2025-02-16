import pdb

from agents.base_agent import BaseAgent
from common.registry import registry
# from rouge import Rouge
import json
import random
import re
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random

def count_tokens(text):
    return len(text.split())

def calculate_similarity(obs1, obs2):
    vectorizer = TfidfVectorizer().fit_transform([obs1, obs2])
    vectors = vectorizer.toarray()
    cos_sim = cosine_similarity(vectors)
    return cos_sim[0, 1]

def calculate_value(memory, w1=1.0, w2=1.0, w3=1.0):
    n = len(memory)
    values = np.zeros(n)
    
    observations = [step[1][1] if len(step) == 4 else step[0][1] for step in memory]  # 观察
    rewards = [step[-1] for step in memory]  # 奖励
    
    mu = (n - 1) / 2
    sigma = n / 6
    boundary_weights = 1 - np.exp(-((np.arange(n) - mu) ** 2) / (2 * sigma ** 2))
    
    values += w1 * boundary_weights
    
    for i in range(n-1):
        similarity = calculate_similarity(observations[i], observations[i+1])
        change_rate = 1 - similarity
        reward_change = rewards[i+1] - rewards[i]
        values[i] += w2 * change_rate
        values[i] += w3 * reward_change

    return values


def select_top_steps(memory, values, m):
    top_indices = np.argsort(values)[-m:]  # 选择值最高的m个步骤
    return [memory[i] for i in sorted(top_indices)]

@registry.register_agent("OurAgent")
class OurAgent(
    BaseAgent):  # the agent should receive goal, state and action, then return the next state
    def __init__(self,
                 llm_model,
                 memory_size=100,
                 # set this to a very large number if you want to keep all history till context length limit
                 examples=[],
                 instruction="",
                 init_prompt_path=None,
                 system_message="You are a helpful assistant.",
                 need_goal=False,
                 check_actions=None,
                 check_inventory=None,
                 use_parser=True,
                 ):
        super().__init__()
        self.use_parser = use_parser
        self.llm_model = llm_model
        self.memory_size = memory_size
        self.goal = None
        self.init_obs = None
        if init_prompt_path is not None:  # load from file
            self.init_prompt_dict = json.load(open(init_prompt_path, 'r'))
            self.instruction = self.init_prompt_dict["instruction"]
            self.examples = self.init_prompt_dict["examples"]
        else:

            self.instruction = instruction
            self.examples = examples

            # self.reset(goal, init_obs)
            self.init_prompt_dict = {
                "examples": examples,
                "instruction": instruction,
                "system_msg": system_message
            }

        self.max_context_length = self.llm_model.context_length
        self.need_goal = need_goal
        self.check_actions = check_actions
        self.check_inventory = check_inventory

        self.example_prompt = None

        if "claude" in self.llm_model.engine:
            self.split = self.llm_model.xml_split
        else:
            self.split = {"example": [""],
                          "text": [""],
                          "rule": [""],
                          "system_msg": [""],
                          "instruction": [""],
                          "goal": [""]}

    def get_example_prompt(self): #return the prompt for an interaction turn
        return self.example_prompt
    
    def log_example_prompt(self, prompt):
        self.example_prompt = prompt

    def reset(self, goal, init_obs, init_act=None):
        self.goal = goal
        self.init_obs = init_obs
        # [action, observation, index, reward]
        self.memory = [[("Action", init_act), ('Observation', self.init_obs), 0, 0]] if init_act \
            else [
            [('Observation', self.init_obs), 0, 0]]  # list of [('State', "xxx"), ('Action', "xxx"), ...]
        self.steps = 0
        self.done = False

    def update(self, action, state):
        self.steps += 1

        # self.memory.append(("Action", action))
        # self.memory.append(("Observation", state))
        reward = 0
        self.memory.append([("Action", action), ("Observation", state), len(self.memory), reward])

    def sample_trajectory(self, history, sampling_n):
        if len(history) < 2:
            return history

        # values = calculate_value(history, w1=1.0, w2=1.0, w3=1.0)
        # sampled_history = select_top_steps(history, values, sampling_n)
        # 确保首尾步骤被保留
        first_step = history[0]
        last_step = history[-1]

        # 排除首尾步骤
        middle_steps = history[1:-1]
        middle_values = calculate_value(middle_steps, w1=1.0, w2=1.0, w3=1.0)
        
        # 计算需要从中间步骤中选择的数量
        m = sampling_n - 2  # 首尾已保留2个
        sampled_middle_steps = select_top_steps(middle_steps, middle_values, m)
        
        # 创建新的history，包括首尾步骤和采样后的中间步骤
        sampled_history = [first_step] + sampled_middle_steps + [last_step]
        # 创建新的history，并将未选择的步骤替换为"Omitted"
        sampled_indexes = set(step[-2] for step in sampled_history)
        full_history = []
        for i in range(len(history)):
            if i in sampled_indexes:
                full_history.append(history[i])
            else:
                index = history[i][-2]
                full_history.append([("Omitted", "Action-Observation pair"), index, 0])
        
        return full_history


    def make_prompt(self, need_goal=False, check_actions="check valid actions", check_inventory="inventory", system_message=''):

        def serialize_history(history):
            res = []
            for item in history:
                index = item[-2]
                item = item[:-2]
                history = history
                for _ in item:
                    res.append(f"[{index}] " + _[0] + ": " + _[1])
            return '\n'.join(res)

        query = ""
        self.instruction += "\nNote: Some action-observation pairs have been executed but are omitted from the prompt due to context length constraints."
        query += self.split["instruction"][0] + self.instruction + self.split["instruction"][-1]

        if isinstance(self.examples, str):
            self.examples = [self.examples]

        if len(self.examples) > 0:
            query += "\nHere are examples:\n" + self.split["example"][0]
            for example in self.examples:
                query += example + "\n"
            query += self.split["example"][-1]
        if need_goal:
            query += self.split["goal"][0] + "You should perform actions to accomplish the goal: " + self.goal + "\n" + \
                     self.split["goal"][-1]
        if check_actions is not None:
            query += "You should use the following commands for help when your action cannot be understood: " + check_actions + "\n"
        if check_inventory is not None:
            query += "You should use the following commands for help when your action cannot be understood: inventory\n"

        # history = self.memory[-self.memory_size:]
        history = self.memory
        original_history = history
        history = self.sample_trajectory(history, self.memory_size)
        # input_prompt = query + "\n".join([item[0] + ": " + item[1] for item in history])
        input_prompt = query + serialize_history(history)

        input_prompt += f"\n[{len(self.memory)}] Action: "

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": input_prompt}
        ]
        num_of_tokens = self.llm_model.num_tokens_from_messages(messages)
        sampling_n = len(original_history)
        if num_of_tokens > self.max_context_length - self.llm_model.max_tokens:
            sampling_n -= 1
            history = self.sample_trajectory(original_history, sampling_n)
            # input_prompt = query + "\n".join([item[0] + ": " + item[1] for item in history])
            input_prompt = query + serialize_history(history)
            # input_prompt += "\nAction: "
            input_prompt += f"\n[{len(self.memory)}] Action: "
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": input_prompt}
            ]
            num_of_tokens = self.llm_model.num_tokens_from_messages(messages)
        print(input_prompt)
        return input_prompt

    def action_parser_for_special_llms(self, action):
        
        '''
        This function is used to parse the action for special llms, e.g. codellama-13b, codellama-34b, llama, lemur, vicuna, etc.
        These llms often struggle to generate the format of the action correctly, so we need to parse the action to make it executable.
        '''
        
        origin_action = action
        if 'action' in action.lower():
            action_temp = action.split('\n')
            for act in action_temp:
                if "next action" in act and ':' in act: # zzh: in Claude will return "Here is the next action to take:"
                    idx = action_temp.index(act)
                    while idx + 1 < len(action_temp):
                        if action_temp[idx + 1]:
                            action = action_temp[idx + 1]
                            break
                        idx += 1
                if act.split(':')[0].lower().endswith('with action input'): # chang: in case parse tool output
                    action = act
                    break
                if 'action' in act.lower() and ':' in act:
                    action_temp = ':'.join(act.split(':')[1:])
                    if action_temp != "":
                        action = action_temp
                        break
                if 'action' in act.lower() and 'is to' in act:
                    action_temp = act.split('is to')[1]
                    if action_temp != "":
                        action = action_temp
                        break
                        
        # if action.strip() == "":
        #     action = origin_action.split('\n')[0]   # temperary comment this line for codellama
        action = action.strip()
        action = action.strip("'/")
        action = action.split('\n')[0]
        return action

    def run(self, init_prompt_dict=None):
        # note that these configs are originally provided when initialized, but you can choose to override them here with parameters
        if init_prompt_dict is not None:
            self.init_prompt_dict = init_prompt_dict
            self.instruction = init_prompt_dict['instruction']
            self.examples = init_prompt_dict['examples']
        system_message = self.init_prompt_dict['system_msg']
        input_prompt = self.make_prompt(need_goal=self.need_goal,
                                        check_actions=self.check_actions,
                                        check_inventory=self.check_inventory,
                                        system_message=system_message)
        
        self.log_example_prompt(input_prompt)

        success, action = self.llm_model.generate(system_message, input_prompt)
        # print('original output', action)
        # print(self.use_parser)
        if success and self.use_parser:
            action = self.action_parser_for_special_llms(action)
            # print('after parse', action)
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
        return cls(llm_model, memory_size, examples, instruction, init_prompt_path, system_message, 
                   need_goal, check_actions, check_inventory, use_parser)