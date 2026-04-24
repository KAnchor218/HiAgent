import abc
import time

class BaseTask:
    def __int__(self):
        super().__int__()

        '''
        initialize llm and agent here for evaluation
        
        self.llm = load_llm(...)
        self.agent = load_agent(...)
        '''
        
    def evaluate_env(self, env_id):
        '''
        evaluate problem {env_id}
        
        '''
        return

    def evaluate(self):
        '''
        evaluate all problems
        
        num_envs = ...
        for id in range(num_envs):
            evaluate_env(id)
            ...
        '''
        return

    def reset_llm_runtime_stats(self):
        llm = getattr(self, "llm", None)
        clear_runtime_stats = getattr(llm, "clear_runtime_stats", None)
        if callable(clear_runtime_stats):
            clear_runtime_stats()

    def get_llm_retry_overhead_time(self):
        llm = getattr(self, "llm", None)
        get_retry_overhead_time = getattr(llm, "get_retry_overhead_time", None)
        if callable(get_retry_overhead_time):
            return get_retry_overhead_time()
        return 0.0

    def get_effective_elapsed_time(self, start_time):
        elapsed_time = time.time() - start_time
        retry_overhead_time = self.get_llm_retry_overhead_time()
        return max(0.0, elapsed_time - retry_overhead_time)

    @classmethod
    def from_config(cls,
                    run_config,
                    llm_config,
                    agent_config,
                    env_config,
                    llm = None  
                    ):
        '''
        Intialize task object from configs
        '''
        return cls()
