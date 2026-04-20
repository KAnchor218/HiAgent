import pdb

from agents.base_agent import BaseAgent
from common.registry import registry
# from rouge import Rouge
import json
import random
import re
import os

from .summarize import TrajectorySummarizer


def extract_numbers(action_string):
    matches = re.findall(r'retrieve\((\d+(?:, \d+)*)\)', action_string)
    if matches:
        return [int(id) for id in matches[0].split(', ')]
    else:
        return []

@registry.register_agent("ContextEfficientAgentV2")
class ContextEfficientAgentV2(
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
        
        self.subgoal_idx = [] # 记录需要重新展开的旧子目标的index，注意此时的index是subgoal_index_list的下标，而不是history的index

    def get_example_prompt(self): #return the prompt for an interaction turn
        return self.example_prompt
    
    def log_example_prompt(self, prompt):
        self.example_prompt = prompt

    def log_example_prompt_subgoal(self, prompt):
        self.example_prompt = prompt
    
    def log_example_prompt_action(self, prompt):
        self.example_prompt = f'```subgoal\n{self.example_prompt}\n```\n```action\n{prompt}\n'

    def reset(self, goal, init_obs, init_act=None):
        self.goal = goal
        self.init_obs = init_obs
        self.memory = [[("Action", init_act), ('Observation', self.init_obs)]] if init_act \
            else [
                [('Observation', self.init_obs)]]  # list of [('State', "xxx"), ('Action', "xxx"), ...]
        self.steps = 0
        self.done = False

    def update(self, action, state):
        '''
        This function is used to update the memory of the agent with the new action and state.
        '''
        self.steps += 1

        # self.memory.append(("Action", action))
        # self.memory.append(("Observation", state))
        self.memory.append([("Action", action), ('Observation', state)])

    def make_prompt(self, need_goal=False, check_actions="check valid actions", check_inventory="inventory", system_message=''):
        def vanilla_serialize_history(history):
            '''
            用来将memory中列表嵌列表形式的历史记录转换成普通字符串的函数。每个子列表中的元素是一个subgoal元组或者(action, observation)对，
            函数会把它们转换成"Action: xxx\nObservation: xxx"的格式，并把所有的子列表连接起来形成最终的历史记录字符串，从而送入LLM。
            
            '''
            res = []
            for item in history:
                for _ in item:
                    res.append( _[0] + ": " + _[1])
            return '\n'.join(res)

        def serialize_history(history):
            '''
            把working memory进行处理并送入vanilla_serialize_history改写成 prompt。
            关键思想是：最后一个子目标保留详细轨迹（可能任务最近一个子任务与当前子任务相关性高），之前已完成的子目标压缩成“编号子目标 + 摘要/终端观察”，以节省上下文长度。
            '''
            self.task = os.environ.get('EVALTASK')
            # if self.task in ['gripper', 'blocksworld']:
            if any([_ in self.task for _ in ['gripper', 'blocksworld']]): # 对于gripper与blocksworld，不使用总结，只压缩，只保留子目标结束时的observation。因为这些环境是完全可观测的，没必要再进行总结了
                summarization = False    # For gripper and blocksworld, set to False.
            else:
                summarization = True
            # ommit_prefix = 'Subgoal is satisfied, and the process is ommited. '
            # ommit_prefix = 'Subgoal is satisfied. ' 
            # locate last subgoal
            subgoal_index_list = [] # 记录subgoal字段在history中的index
            keep_subgoal_index_list = [_-1 for _ in self.subgoal_idx] # 将subgoal_idx中的index转换成从0开始的index，并记录在keep_subgoal_index_list里，表示这些子目标需要保留详细轨迹信息，不进行压缩
            for i in range(0, len(history)): # 把memory中的subgoal找出来并记录subgoal字段在history中的index，存在subgoal_index_list里
                item = history[i]
                if item[0][0] == 'Subgoal':
                    subgoal_index_list.append(i)
            if len(subgoal_index_list) <= 1: # 如果当前历史中只有 0 个或 1 个子目标，就直接用普通方式序列化历史，不走压缩逻辑（不需要压缩），直接return。
                return vanilla_serialize_history(history)
            final_subgoal = subgoal_index_list[-1]
            new_history = history[:subgoal_index_list[0]] # 把第一个子目标之前的历史记录原封不动地保留下来。最终用于存储执行目前子目标时之前的working memory
            for i in range(0, len(subgoal_index_list)-1): # 依次遍历每一个子目标（除了最后一个子目标），对其memory chunk进行操作。对于需要展开的进行展开(其实就是不压缩)，对于不需要展开的进行压缩
                if i in keep_subgoal_index_list: # 遍历每一个subgoal，看是否需要展开
                    new_history += history[subgoal_index_list[i]:subgoal_index_list[i+1]] # 需要展开则把这个子目标对应的完整轨迹（从subgoal开始到下一个subgoal之前的所有记录）都保留下来加入new_history
                    continue
                index = subgoal_index_list[i] # 子目标不需要展开时，记录其index
                obs_index = subgoal_index_list[i+1] - 1 # 下一个子目标在history中的index减1就是当前子目标对应的最后一个observation的index
                # 在压缩时提前判断是否要进行总结
                if not summarization:   # No summarization. Under full observable environment；若不启用summarization
                    subgoal = history[index][0] # 取出不需要展开的subgoal元组
                    _ = subgoal[0]
                    subgoal = (f'{i+1} {_}', subgoal[1]) # 重建元组，f外面的括号代表元组，把角色名从 "Subgoal" 改为带编号的 "1 Subgoal"、"2 Subgoal" 等（编号对应于subgoal在history中的index），subgoal的内容subgoal[1]不变
                    # new_history.append([subgoal, ("Observation", ommit_prefix + history[obs_index][1][1])])
                    new_history.append([subgoal, ("Observation", history[obs_index][1][1])]) # 不进行总结时，把当前子目标memory压缩为[编号子目标, 最终Observation]。最终Observation即这个子目标的最后一个observation的内容
                else:   # Using the summarizer
                    summarizer = TrajectorySummarizer(self.llm_model) # 对于需要对memory chunk进行总结的，调用TrajectorySummarizer来生成摘要。summarizer是一个TrajectorySummarizer类实例
                    subgoal = history[index][0] # 需要压缩的subgoal对应的subgoal元组
                    trajectory = history[index+1:obs_index+1] # 子目标对应的trajectory（双重列表）
                    # remove check valid actions
                    trajectory = [pair for pair in trajectory if pair[0][0] != 'Action' or 'check valid' not in pair[0][1]] 
                    ''' 
                    对trajectory中的action进行过滤，把包含"check valid"的action过滤掉，
                    因为这些action是用来检查动作合法性的，不是实际执行的动作，在生成摘要前先把它们剔除，避免干扰 LLM的摘要质量，同时节省 token
                    pair[0][0] != 'Action'理论上不会触发，是一个防御性判断
                    '''
                    # if len(trajectory) == 1:
                    #     pair = trajectory[0]
                    #     if pair[0][0] == 'Observation':
                    #         summary = pair[0][1]
                    #     else:
                    #         summary = pair[1][1]
                    # else:
                    summary = summarizer.generate_summary([trajectory], [subgoal])[0]
                    # reformat subgoal
                    subgoal = history[index][0]
                    _ = subgoal[0]
                    subgoal = (f'{i+1} {_}', subgoal[1]) # 与not summarize时一样，把角色名改成带编号的，内容不变
                    new_history.append([subgoal, ("Observation", summary)])  # ????

            # new_history += history[final_subgoal:]
            # add number of last subgoal
            # - = []
            subgoal = history[final_subgoal][0]
            _ = subgoal[0]
            subgoal = (f'{len(subgoal_index_list)} {_}', subgoal[1]) # 最后一个子目标同样改成带编号的形式
            _ = [[subgoal]] + history[final_subgoal+1:] # 最后一个子目标及其之后的历史记录原封不动地保留下来加入new_history，subgoal放在前面表示在subgoal元组与action-observation对合并时也在前面
            new_history += _ # 把最后一个子目标也加入new_history
            return vanilla_serialize_history(new_history)

        query = ""
        # _ = "\nNote: A subgoal is a milestone goal that you need to complete in order to achieve the final goal, while an action is a specific step executed in the environment. When there is an unfinished subgoal, you need to output an action to continue completing this subgoal in the following format: \"Action: {action}\". When there is no current subgoal or you believe the previous subgoal has been completed (based on past actions and observations), you need to output the next subgoal to be completed in the following format: \"Subgoal: {subgoal}\". You cannot output two subgoals consecutively."
        # _ = "\nNote: A subgoal is a milestone goal that you need to complete in order to achieve the final goal. When there is an unfinished subgoal, you need to ground the given subgoal to corresponding executable actions for solving the given task in the following format: \"Action: {action}\". When there is no current subgoal or you believe the previous subgoal has been completed (based on past actions and observations), you need to output the next subgoal to be completed in the following format: \"Subgoal: {subgoal}\". You cannot output two subgoals consecutively."
        # _ = "\nNote: A subgoal is a milestone goal that you need to complete in order to achieve the final goal. When there is an unfinished subgoal, you need to ground the given subgoal to corresponding executable actions for solving the given task in the following format: \"Action: {action}\". When there is no current subgoal or you believe the previous subgoal has been completed (based on past actions and observations), you need to output the next subgoal to be completed and its first action in the following format: \"Subgoal: {subgoal}\\nAction: {action}\". You cannot output two subgoals consecutively."
        # _ = "\nNote: A subgoal is a milestone goal that you need to complete in order to achieve the final goal. When there is an unfinished subgoal, you need to ground the given subgoal to corresponding executable actions for solving the given task in the following format: \"Action: {action}\". When there is no current subgoal or you believe the previous subgoal has been completed (based on past actions and observations), you need to output the next subgoal to be completed and its first action in the following format: \"Subgoal: {subgoal}\\nAction: {action}\". You cannot output two subgoals consecutively. Detailed trajectory information (action-observation pair) of previously satisfied subgoals will be hidden for context efficiency. If you believe that the detailed trajectory information of a particular subgoal is crucial for the current subgoal, you can use Action: \"retrieve(subgoal_id)\" to obtain the detailed trajectory information. You should use this method judiciously for token efficiency."
        # remove the constrains of retrieve
        #         _ = """
        # Note: A subgoal is a milestone goal that you need to complete in order to achieve the final goal. 
        # When there is an unfinished subgoal, you need to ground the given subgoal to corresponding executable actions for solving the given task in the following format: \"Action: {action}\". 
        # When there is no current subgoal or you believe the previous subgoal has been completed (based on past actions and observations), you need to output the next subgoal to be completed and its first action in the following format: \"Subgoal: {subgoal}\\nAction: {action}\". 
        # Hints:
        # 1. You cannot output two subgoals consecutively. 
        # 2. Subgoal must be one line of text and does not print any newline characters. Detailed trajectory information (action-observation pair) of previously satisfied subgoals will be hidden for context efficiency. If you believe that the detailed trajectory information of a particular subgoal is crucial for the current subgoal, you can use Action: \"retrieve(subgoal_id_1, subgoal_id_2, ...)\" to obtain the detailed trajectory information.
        # """
        _ = """
Note: A subgoal is a milestone goal that you need to complete in order to achieve the final goal. 
When there is an unfinished subgoal, you need to ground the given subgoal to corresponding executable actions for solving the given task in the following format: \"Action: {action}\". 
When there is no current subgoal or you believe the previous subgoal has been completed (based on past actions and observations), you need to output the next subgoal to be completed and its first action in the following format: \"Subgoal: {subgoal}\\nAction: {action}\". 
Instructions:
1. You cannot output two subgoals consecutively. 
2. Subgoal must be one line of text and does not print any newline characters. 
3. Each subgoal must be followed by the execution of at least one valid action. If the current action fails, you need to execute "check valid actions" to get a list of valid actions and select one from the list.
4. Detailed trajectory information (action-observation pair) of previously satisfied subgoals will be hidden for context efficiency. If you believe that the detailed trajectory information of a particular subgoal is crucial for the current subgoal, you can use Action: \"retrieve(subgoal_id_1, subgoal_id_2, ...)\" to obtain the detailed trajectory information.
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
            query += self.split["goal"][0] + "You should perform actions to accomplish the goal: " + self.goal + "\n" + \
                     self.split["goal"][-1]
        if check_actions is not None:
            query += "You should use the following commands for help when your action cannot be understood: " + check_actions + "\n"
        if check_inventory is not None:
            query += "You should use the following commands for help when your action cannot be understood: inventory\n"

        history = self.memory[-self.memory_size:] # 根据上下文窗口进行截断，只是用一部分memory
        input_prompt = query + serialize_history(history)

        input_prompt += "\nAction: " if self.memory[-1][0][0] == 'Subgoal' else ""  # 如果当前最新的一条记录是subgoal字段，则加上Action引导大模型进行输出（防御性措施，约等于多余）

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": input_prompt}
        ]
        num_of_tokens = self.llm_model.num_tokens_from_messages(messages)
        while num_of_tokens > self.max_context_length - self.llm_model.max_tokens:
            history = history[1:] # 再进行截断，每次截断掉最早的一条记录，直到满足上下文长度限制
            input_prompt = query + serialize_history(history)
            # input_prompt += "\nAction: "
            input_prompt += "\nAction: " if self.memory[-1][0][0] == 'Subgoal' else ""
            # input_prompt += "\nPlease enter your action:"
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": input_prompt}
            ]
            num_of_tokens = self.llm_model.num_tokens_from_messages(messages)
        print(f'------------[Prompt Start]-----------\n{input_prompt}\n----------[Prompt END]------------')
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
        
        # 记录输入LLM的提示词，用于日志
        self.log_example_prompt(input_prompt)
        
        success, action = self.llm_model.generate(system_message, input_prompt) # success代表的是LLM是否调用成功并返回了结果
        print(f'-------------GPT Response---------\n{action}\n---------------[END]------------')
        # 让LLM生成subgoal
        if success:
            # action = action.split('\n')[0]
            is_action = 'Subgoal' not in action # 判断生成的是subgoal还是action
            # is_action = action.startswith('Action') or 'Subgoal' not in action
            if not is_action: # 生成的是subgoal，需要把subgoal加入memory，并且把生成的subgoal从action中去掉，剩下的部分才是action
                # match = re.search(r"Subgoal:(.*?)(?=\n)", action)
                # if match:
                #     subgoal = match.group(1).strip()
                # action = action.replace(f'Subgoal: {subgoal}', '')
                subgoal = action.split('\n')[0]
                subgoal = subgoal.replace('Subgoal:', '')
                self.subgoal_idx = []
                self.memory.append([("Subgoal", subgoal)]) # 生成的subgoal加入memory
                action = '\n'.join(action.split('\n')[1:]) # 去掉subgoal之后的内容，包括action和retrieve(
            # print('original output', action)
            # print(self.use_parser)
            # if is_action:
            if self.use_parser: # 对于一些特殊的llm（如codellama系列、lemur、vicuna等），需要对其输出的action进行特殊处理，才能得到正确的动作指令
                action = self.action_parser_for_special_llms(action)
                # print('after parse', action)    
            if 'retrieve(' in action.lower():   # retrieve function is called
                action = action.lower()
                numbers = extract_numbers(action)
                # self.subgoal_idx.append(number)
                self.subgoal_idx += numbers
                return self.run(init_prompt_dict)  # 获取历史详细信息之后再进行一次决策，注意这里的递归调用会生成新的prompt并调用LLM，所以需要在make_prompt函数中根据self.subgoal_idx来判断哪些子目标需要展开详细历史，哪些子目标需要压缩成摘要或者直接丢掉历史记录
            # else:   # subgoal
            #     subgoal = action.replace('Subgoal:', '')
            #     self.memory.append([("Subgoal", subgoal)])
            #     return self.run(init_prompt_dict)
        return success, action # 没有subgoal时直接返回动作即可，有subgoal时，先对subgoal进行处理

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
