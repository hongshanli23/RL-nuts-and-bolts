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

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_


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

def evalute_critic(agent)->float:
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
    
    writer = Logger(h['log_dir'])
    
    agent = Agent(model, h)
    sampler = TrajectorySampler(env, agent)
    
    for i in range(h['n_iters']):
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
            clip_grad_norm_(agent.model.parameters(), h['clip_grad_norm'])        
        agent.optimizer.step()
        
        # logging
        writer.add_scalar('loss', float(loss), i)
        writer.add_scalar('p_loss', float(p_loss), i)
        writer.add_scalar('v_loss', float(v_loss), i)
        writer.add_scalar('entropy', float(entropy), i)
        writer.add_scalar('adv', float(torch.mean(adv)), i)
        writer.add_scalar('log_pi_a', float(torch.mean(log_pi_a)), i)
        writer.add_scalar('ma_rew', float(reward_tracker.query().mean()), i)
        
        if h['evalute_critic']:
            v = evalute_critic(agent)
            writer.add_scalar('init_state_estimate', v.mean(), i)

        if i % h['ckpt_interval'] == 0:
            agent.save_ckpt(i)
    
    agent.save_ckpt(i)
    writer.save()
    return agent


    
