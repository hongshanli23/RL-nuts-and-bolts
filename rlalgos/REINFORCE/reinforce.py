# REINFORCE

from contextlib import contextmanager
from collections import deque
import numpy as np
import time
import os
import gym
import pprint as pp
import pickle
import sys

import rlkits.utils.logger as logger
from rlkits.env_wrappers import AutoReset

from rlkits.sampler import SimpleTrajectorySampler 
from rlkits.sampler import shuffle_experience
from rlkits.policies import REINFORCEPolicy
import rlkits.utils as U
from rlkits.utils import colorize
import rlkits.utils.logger as logger
from rlkits.utils.math import explained_variance, safemean


import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from typing import Dict
from ipdb import set_trace

def compute_loss(policy, trajectory):
    """Compute policy loss along a trajectory

    Args:
        policy (Policy): current policy
        trajectory (Dict): trajectory sampled from the current policy

    Returns:
        torch.Tensor: policy loss
    """
    obs = torch.from_numpy(trajectory['obs'])
    
    # policy as distribution over action space
    dist = policy.dist(policy.model(obs))
    if dist is None:
        logger.log('Got Nan -- Bad')
        policy.save_ckpt('broken')
        args = {
            "trajectory":trajectory
        }
        with open(os.path.join(policy.ckpt_dir, 'local.pkl'), 'wb') as f:
            pickle.dump(args, f)
        sys.exit()
    
    actions = torch.from_numpy(trajectory['actions'])
    log_prob = dist.log_prob(actions)
    Q = torch.from_numpy(trajectory['Q'])
    
    if len(log_prob.shape) > 1:
        log_prob = log_prob.squeeze(dim=1)
    assert log_prob.shape==Q.shape, f"log_prob shape: \
        {log_prob.shape} Q shape: {Q.shape}"
    
    # pytorch optimizer only does gradient descent
    # so our objective becomes minimize -J(\theta)
    policy_loss = -(log_prob * Q).mean()
    return policy_loss
    
def REINFORCE(*,
              env_name,
              nsteps,
              total_timesteps,
              gamma,
              pi_lr,
              log_interval,
              max_grad_norm,
              ckpt_dir,
              **network_kwargs
              ):
    """
    REINFORCE algorithm

    Args:
        env_name (str): name of the gym env
        nsteps (int): length of a trajectory to sample
            for each policy update 
        total_timesteps (int): total number of frames to 
            train
        gamma (float): discount factor 
        pi_lr (float): learning rate
        log_interval (int): log and print per $log_interval
            training steps 
        max_grad_norm (float): gradient of parameters
            in each layer is clipped so that the norm
            for each layer is less than or equal to 
            $max_grad_norm
        ckpt_dir (str): directory to save the checkpoint 

    Returns:
        None
    """
        
    # log and ckpts    
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    logger.configure(dir=ckpt_dir)

    # env 
    def make_env():
        env = gym.make(env_name)
        env = AutoReset(env)
        return env
    
    env = make_env() 
    ob_space = env.observation_space
    ac_space = env.action_space
    
    # init a policy
    policy = REINFORCEPolicy(ob_space=ob_space,
                    ac_space=ac_space, 
                    ckpt_dir=ckpt_dir, 
                    **network_kwargs)
    
    optimizer = optim.Adam(policy.model.parameters(), lr=pi_lr)

    # trajectory sampler
    sampler = SimpleTrajectorySampler(env, policy, nsteps)

    # moving average of total rewards in the last 10 episodes
    # an episode is defined as a sequence of policy
    # execution until env terminates
    # for some env, it might never terminate
    rolling_buf_episode_rets = deque(maxlen=10)

    # number of param updates
    nupdates = total_timesteps // nsteps

    # performance 
    best_ret=np.float('-inf')
    start = time.perf_counter()
    for update in range(1, nupdates+1):
        # timestamp at the beginining of the training
        tstart = time.perf_counter()
        
        # sample a trajectory of length nsteps
        trajectory = sampler() 
        
        # when computing reinforcement offset, it helps
        # to reduce variance by normalizing the reward
        # collected at each time step
        # trajectory['rews'] = (
        #    trajectory['rews'] - np.mean(trajectory['rews'])
        #) / np.std(trajectory['rews'])
        
        Q = np.zeros(nsteps, dtype=np.float32)
        future_rews = 0.0
        for t in reversed(range(0, len(trajectory))):
            not_done = not trajectory['dones'][t]
            Q[t] = trajectory['rews'][t] + gamma * not_done * future_rews
            future_rews = Q[t]
        trajectory['Q'] = Q
            
        # use forward view to update total rewards
        for t in range(0, len(trajectory)):
            not_done = not trajectory['dones'][t]
            if not_done and rolling_buf_episode_rets:
                # add reward to the last episode
                rolling_buf_episode_rets[-1]+=trajectory['rews'][t]
            else:
                rolling_buf_episode_rets.append(trajectory['rews'][t])
        
        # compute policy loss
        loss = compute_loss(policy, trajectory)
        
        optimizer.zero_grad()
        loss.backward()
        # clip the gradient to make the training more stable
        clip_grad_norm_(policy.model.parameters(), 
                        max_norm=max_grad_norm)
        optimizer.step()
        
        # timestamp when finish one step of training
        tnow = time.perf_counter()
        
        # logging
        if update % log_interval == 0 or update == 1:
            # performance
            # frames / sec
            fps = nsteps // (tnow - tstart)
            logger.record_tabular('fps', fps)
            logger.record_tabular('policy_loss', 
                                  loss.detach().numpy())
            # moving average of episodic returns
            ret = safemean(rolling_buf_episode_rets)
            logger.record_tabular('ma_ep_ret', ret)
            # average reward per step
            logger.record_tabular('mean_rew_step', 
                                 np.mean(trajectory['rews'])
                                 )
            # save the best ckpt
            if ret != np.nan and ret > best_ret:
                best_ret = ret
                policy.save_ckpt('best', optimizer)


            logger.dump_tabular()
        #set_trace()
    
    policy.save_ckpt('final', optimizer)
    end = time.perf_counter()
    logger.log(f"Total time elapsed: {end - start}")
    return

if __name__ == '__main__':
    REINFORCE(
        env_name='CartPole-v0',
        nsteps=32,
        total_timesteps=int(1e6),
        gamma=0.99,
        pi_lr=1e-4,
        log_interval=1000,
        max_grad_norm=0.1,
        ckpt_dir="/tmp",
        hidden_layers=[64, 128]
    )
    
                                
            
            
        
         


            
            
            
        
        
        
