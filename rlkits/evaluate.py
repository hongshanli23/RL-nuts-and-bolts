import numpy as np
import gym
import torch

from rlkits.env_batch import EnvBatch
from rlkits.env_batch import SingleEnvBatch, ParallelEnvBatch
from rlkits.env_batch import ParallelEnvBatch
from rlkits.env_wrappers import AutoReset, StartWithRandomActions

class ParallelEnvPolicyEvaluator:
    """Evaluate a policy with parallel envs"""
    def __init__(self, env_name, policy, nenvs=4):
        
        def make_env():
            env = gym.make(env_name)
            env = AutoReset(env)
            env = StartWithRandomActions(env, max_random_actions=5)
            return env
        

def evaluate_policy(env_name, policy):
    """Evaluate a policy with parallel envs"""
    def make_env():
        env = gym.make(env_name)
        env = AutoReset(env)
        env = StartWithRandomActions(env, max_random_actions=5)
        return env
        
    env = ParallelEnvBatch(make_env, nenvs=1)
    rews = [] # rewards from 
    ob = env.reset()
    done = False
    
    while not done:
        ob = torch.from_numpy(ob.astype(np.float32))
        action = policy.step(ob)
        ob, rew, done, _ = env.step(action)
        rews.append(rew)
    
    env.close()
    return rews
        