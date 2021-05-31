# train an agent with A2C 
from algo import Agent, estimate_Q
from algo import TrajectorySampler, compute_loss
from utils import AverageEpisodicRewardTracker
from utils import Logger

from env_batch import ParallelEnvBatch
from env_wrappers import AutoReset, StartWithRandomActions
from model import MLPSingleArch

import numpy as np
import gym
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_

import rlkits.utils.logger as logger
from rlkits.utils.math import explained_variance


def evaluate(env, agent, max_steps=1000)-> int:
    """Evaluate the performance of the agent in an the env

    Args:
        env: an instance of gym env
        agent: an instance of agent
        max_steps: max steps to play

    Return 
        total reward
    """
    state = env.reset()
    total_reward = 0.0
    for _ in range(max_steps):
        action = agent.take_action(state, greedy=True)
        nx_state, reward, done, _ =  env.step(action[0])
        total_reward += reward
        if done:
            break
        state = nx_state
    return total_reward

def evaluate_critic(agent)->float:
    env = gym.make('CartPole-v0').unwrapped
    env.seed(10)
    return agent.get_value(env.reset())

def make_env():
    """Used to create parallel envs in subprocesses"""
    env = gym.make('CartPole-v0').unwrapped
    env.reset()

    env = AutoReset(env)
    env = StartWithRandomActions(env, max_random_actions=5)
    return env


def a2c(h, decay_fn=None):
    """Train an agent with A2C algorithm"""
    reward_tracker = AverageEpisodicRewardTracker(h['nenvs'])
    env = ParallelEnvBatch(make_env, nenvs=h['nenvs'])

    model = MLPSingleArch(
        input_dim = env.observation_space.shape[0],
        n_actions=env.action_space.n
    )
    
    #writer = Logger(h['log_dir'])
    logger.configure(h['log_dir'])
    
    agent = Agent(model, h)
    sampler = TrajectorySampler(env, agent)
    
    start = time.perf_counter()
    for i in range(1, h['n_iters']+1):
        tstart = time.perf_counter()
        
        trajectory = sampler(h['nsteps'])
        Q = estimate_Q(agent, trajectory, h['gamma'], h['nenvs'])
        
        batch_reward = np.zeros((h['nsteps'], h['nenvs']), dtype=np.float32)
        batch_done = np.zeros((h['nsteps'], h['nenvs']), dtype=np.bool)
        
        for j, e in enumerate(trajectory):
            batch_reward[j] = e.reward
            batch_done[j] = e.done
            
        reward_tracker.update(batch_reward, batch_done)
        
        # compute loss
        ld = compute_loss(trajectory, Q, h['device'])
        
        p_loss, v_loss, entropy = ld['p_loss'], ld['v_loss'], ld['entropy']
        adv, log_pi_a = ld['adv'], ld['log_pi_a']
        
        if decay_fn and 'entropy_coef' in decay_fn:
            h['entropy_coef'] = decay_fn['entropy_coef'](h['entropy_coef'], i)
            
        loss = h['p_coef']*p_loss + h['v_coef']*v_loss - \
        h['entropy_coef']*entropy

        # optimizing
        agent.optimizer.zero_grad()
        loss.backward()
        
        if h['clip_grad_norm']:
            clip_grad_norm_(agent.model.parameters(), 0.1)        
        agent.optimizer.step()
        
        tnow = time.perf_counter()
        fps = int(h['nenvs']*len(trajectory) / (tnow - tstart))
        # logging
        if i % h['ckpt_interval'] == 0:
            logger.record_tabular('FPS', fps)
            logger.record_tabular('iter/niters', f"{i}/{h['n_iters']}")
            logger.record_tabular('average_param', agent.model_weight())
            logger.record_tabular('policy_loss', p_loss.detach().numpy())
            logger.record_tabular('value_loss', v_loss.detach().numpy())
            logger.record_tabular('entropy', entropy.detach().numpy())
            
            # aggregate V
            vpreds = [e.v.detach().numpy() for e in trajectory]
            vpreds = np.array(vpreds)
            vpreds = np.squeeze(vpreds, axis=2)
            assert vpreds.shape == Q.shape, f'vpreds shape: {vpreds.shape}, Q shape: {Q.shape}'
            vpreds = vpreds.transpose(1, 0).ravel()
            Q = Q.transpose(1, 0).ravel()
            
            logger.record_tabular('vpreds', vpreds.mean())
            logger.record_tabular('Q', Q.mean())
            
            vqdiff = np.mean((Q - vpreds)**2)
            logger.record_tabular('vqdiff', vqdiff)
            
            ev = explained_variance(vpreds, Q)
            logger.record_tabular('explained_var', ev)
            
            ma_rew = reward_tracker.query().mean()
            logger.record_tabular('ma_rew', ma_rew)
            
            if h['evaluate_critic']:
                logger.record_tabular('init_est', evaluate_critic(agent).mean())
            logger.dump_tabular()
    
    end = time.perf_counter()
    logger.log(f"Total training time: {end - start}")
    agent.save_ckpt(i)
    env.close()
    return agent



if __name__ == '__main__':
    h = {
        "n_iters": 10000,
        "ckpt_interval": 10,
        "nenvs":16,
        "nsteps" : 32, # length of trajectory
        "entropy_coef" : 0.01, #1e-4, 
        "learning_rate": 1e-4,
        "p_coef": 1.0,
        "v_coef":1.0,
        "gamma": 0.99,
        "device": "cpu",
        "clip_grad_norm": True,
        "evaluate_critic": True,
        "ckpt_dir": '/home/ubuntu/tmp/A2C/ckpt/debug/1', # to be changed 
        "log_dir": '/home/ubuntu/tmp/A2C/log/debug/1' # to be changed
    }
    
    a2c(h)
    
