from .vanilla_agent import VanillaAgent
from .react_agent import ReactAgent
from .custom_react import CustomReactAgent
from common.registry import registry
from .ours_agent import OurAgent
from .cme_final import ContextEfficientAgentV2

# __all__ 用于定义当使用 'from agents import *' 时会被导入的模块成员列表。
__all__ = ["VanillaAgent", "ReactAgent", "CustomReactAgent", "OurAgent", "ContextEfficientAgentV2"]


def load_agent(name, config, llm_model):
    agent = registry.get_agent_class(name).from_config(llm_model, config)
    return agent
