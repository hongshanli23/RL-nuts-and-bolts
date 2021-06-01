# A2C
# Use a single model for actor-critic agent
# temp boiler-plate code. will go away soon

from contextlib import contextmanager
from collections import deque
import numpy as np
import time
import os
import gym
import pprint as pp
import pickle
import sys


from rlkits.sampler import ParallelEnvTrajectorySampler
from rlkits.sampler import estimate_Q
from rlkits.sampler import aggregate_experience
from rlkits.policies import PolicyWithValueSingleModel
import rlkits.utils as U
from rlkits.utils import colorize
import rlkits.utils.logger as logger
from rlkits.utils.math import explained_variance

import torch 
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_


def compute_loss(pi, trajectory, log_dir):
    """Compute loss for policy and value net"""
    obs = trajectory['obs']
    obs = torch.from_numpy(obs)
    piparams, vpreds = pi.model(obs)
    dist = pi.dist(piparams)
    if dist is None:
        logger.log('Got Nan -- Bad')
        pi.save_ckpt()
        args = {
            "trajectory":trajectory
        }
        with open(os.path.join(log_dir, 'local.pkl'), 'wb') as f:
            pickle.dump(args, f)
        sys.exit()
    
    actions = torch.from_numpy(trajectory['actions'])
    log_prob = dist.log_prob(actions)
    adv = torch.from_numpy(trajectory['adv'])
    if len(log_prob.shape) > 1:
        log_prob = log_prob.squeeze(dim=1)
    assert log_prob.shape == adv.shape, f"log_prob shape: {log_prob.shape}, adv shape : {adv.shape}"
    pi_loss = -log_prob * adv 
    Q = torch.from_numpy(trajectory['Q'])
    if len(vpreds.shape) > 1:
        vpreds = vpreds.squeeze(dim=1)
    assert vpreds.shape == Q.shape, f"vpreds shape: {vpreds.shape}, Q shape : {Q.shape}"
    v_loss = F.mse_loss(vpreds, Q)
    return {
        "pi_loss" : pi_loss.mean(),
        "v_loss"  : v_loss.mean(),
        "entropy" : dist.entropy().mean()
    }


def one_step_loss(pi, s, a, v, log_pi_a, Q):
    """
    s: one state (parallel)
    a: action on the state (parallel)

    v: prediction from value net
        used to check if it agrees with the 
        prediction done in this step with graph 
    log_pi_a: log probabbility on the action
    Q: Q(s,a)
    """
    assert s.ndim == 2, s.ndim
    assert a.ndim == 1, a.ndim
    assert v.ndim == 1, v.ndim
    assert log_pi_a.ndim == 1, log_pi_a.ndim
    assert Q.ndim == 1, Q.ndim

    s = torch.from_numpy(s)
    piparams, vpreds = pi.model(s)

    v = torch.from_numpy(v).unsqueeze(dim=1)
    assert torch.equal(v, vpreds)

    dist = pi.dist(piparams)
    
    if a.dtype == np.int64: # discrete 
        a = torch.from_numpy(a)
    elif a.dtype == np.float32: # continuous
        a = torch.from_numpy(a).unsqueeze(dim=1)

    log_prob = dist.log_prob(a)

    # sampled from env
    log_pi_a = torch.from_numpy(log_pi_a)#.unsqueeze(dim=1)
    assert torch.equal(log_pi_a, log_prob), f"\n{log_pi_a}, \n{log_prob}"
    
    
    Q = torch.from_numpy(Q)
    v = v.squeeze(dim=1)
    assert Q.shape == v.shape, f"\n{Q.shape}, \n{v.shape}"
    adv = Q - v
    
    if log_prob.ndim == 2:
        log_prob = log_prob.squeeze(dim=1)
        
    assert log_prob.shape == adv.shape, f"{log_prob.shape}, {adv.shape}"
    
    entropy = dist.entropy()

    pi_loss = torch.mean(-log_prob * adv)

    # loss on v
    v_loss = F.mse_loss(vpreds.squeeze(dim=1), Q)

    return {
        "pi_loss" : pi_loss,
        "v_loss"  : v_loss,
        "entropy" : entropy.mean()
        }

def compute_loss_v2(pi, trajectory, log_dir):
    """
    compute loss on individual experience
    do forward pass on experience sampled 
    from parallel env
    """
    
    # def one_step_loss(pi, s, a, v, log_pi_a, Q):
    obs = trajectory['obs']
    Qs = trajectory['Q']
    acs = trajectory['actions']
    vs = trajectory['vpreds']
    logps = trajectory['log_prob']
    
    losses = {}
    for t in range(len(obs)):
        one_step_loss_dict = one_step_loss(
            pi, s=obs[t], a=acs[t], v=vs[t], 
            log_pi_a = logps[t], Q=Qs[t])
        for k, v in one_step_loss_dict.items():
            losses[k] = losses.get(k, 0.) + v

    for k in losses:
        losses[k] /= len(obs)
    return losses


def A2C(
    env, 
    nsteps,
    total_timesteps,
    pi_lr,
    ent_coef,
    log_interval,
    max_grad_norm,
    reward_transform,
    log_dir,
    ckpt_dir, 
    **network_kwargs
):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    logger.configure(dir=log_dir) 

    ob_space = env.observation_space
    ac_space = env.action_space

    pi = PolicyWithValueSingleModel(
        ob_space=ob_space,
        ac_space=ac_space, 
         ckpt_dir=ckpt_dir,
        **network_kwargs)
    
    optimizer = optim.Adam(pi.model.parameters(),lr=pi_lr)
 

    sampler = ParallelEnvTrajectorySampler(env, pi, nsteps, 
        reward_transform=reward_transform) # hard-coded for pendulum
    
    # moving average of last 10 episode returns
    rolling_buf_episode_rets = deque(maxlen=100) 
    
    # moving average of last 10 episode length
    rolling_buf_episode_lens = deque(maxlen=100)  
    
    
    nframes = env.nenvs * nsteps # number of frames processed by update iter
    nupdates = total_timesteps // nframes
    
    start = time.perf_counter()
    
    for update in range(1, nupdates+1):
        tstart = time.perf_counter()
        
        trajectory = sampler(callback=estimate_Q)
        
        # aggregate exps from parallel envs
        for k, v in trajectory.items():
            if isinstance(v, np.ndarray):
                trajectory[k] = aggregate_experience(v)
                
        if update == nupdates:
            with open(os.path.join(log_dir, 'trajectory.pkl'), 'wb') as f:
                pickle.dump(trajectory, f)
                
        
        adv = trajectory['Q'] - trajectory['vpreds']
        #trajectory['adv'] = adv
        trajectory['adv'] = (adv - adv.mean())/adv.std()
        
        
        losses = compute_loss(pi=pi,
                           trajectory=trajectory,
                           log_dir=log_dir)
        
        frac = 1.0 - (update - 1.0)/nupdates
        loss = losses['pi_loss'] + losses['v_loss'] \
            - ent_coef * losses['entropy']
        #loss = losses['v_loss']
        
        optimizer.zero_grad()
        loss.backward()
        
        clip_grad_norm_(pi.model.parameters(), max_norm=max_grad_norm)
    
        
        optimizer.step()

        
        tnow = time.perf_counter()
        fps = int(nframes / (tnow - tstart)) # frames per seconds
        
        # logging
        if update % log_interval == 0 or update==1:
            
            logger.record_tabular('iteration/nupdates', 
                                  f"{update}/{nupdates}")
            logger.record_tabular('frac', frac)
            logger.record_tabular('policy_loss', 
                                 losses['pi_loss'].detach().numpy())
            logger.record_tabular('value_loss', 
                                 losses['v_loss'].detach().numpy())
            logger.record_tabular('entropy', 
                                 losses['entropy'].detach().numpy())
            
            for ep_rets in trajectory['ep_rets']:
                rolling_buf_episode_rets.extend(ep_rets)

            for ep_lens in trajectory['ep_lens']:
                rolling_buf_episode_lens.extend(ep_lens)

            
            # explained variance
            ev = explained_variance(trajectory['vpreds'], trajectory['Q'])
            logger.record_tabular('explained_variance', ev)


            w = pi.average_weight()
            logger.record_tabular('average_param', w.numpy())
 
            vqdiff = np.mean((trajectory['Q'] - trajectory['vpreds'])**2)
            
            logger.record_tabular('vqdiff', vqdiff)
            logger.record_tabular('Q', np.mean(trajectory['Q']))
            logger.record_tabular('vpreds', np.mean(trajectory['vpreds']))
            
            logger.record_tabular('FPS', fps)
            logger.record_tabular('ma_ep_ret', 
                                  safemean(rolling_buf_episode_rets))
            logger.record_tabular('ma_ep_len',
                                  safemean(rolling_buf_episode_lens))
            logger.record_tabular('mean_step_rew', 
                                  safemean(trajectory['rews']))
            logger.dump_tabular()
            pi.save_ckpt()
            torch.save(optimizer, os.path.join(ckpt_dir, 'optim.pth'))
            
    end = time.perf_counter()
    logger.log(f"Total training time: {end - start}")
    return


def safemean(l):
    return np.nan if len(l) == 0 else np.mean(l)


if __name__ == '__main__':
    from rlkits.env_batch import ParallelEnvBatch
    from rlkits.env_wrappers import AutoReset, StartWithRandomActions
    
    def make_env():
        env = gym.make('CartPole-v0').unwrapped
        env = AutoReset(env)
        env = StartWithRandomActions(env, max_random_actions=5)
        return env
    
    nenvs = 16
    env=ParallelEnvBatch(make_env, nenvs=nenvs)
    
    A2C(
        env=env,
        nsteps=32,
        total_timesteps=32*nenvs*10000,
        pi_lr=1e-4,
        ent_coef=1e-2,
        log_interval=10,
        max_grad_norm=0.1,
        reward_transform=None,
        log_dir='/home/ubuntu/tmp/log/debug/9',
        ckpt_dir='/home/ubuntu/tmp/log/debug/9',
        hidden_layers=[256, 512],
        activation=torch.nn.ReLU
    )
    
