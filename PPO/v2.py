# PPO 

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
from rlkits.policies import PolicyWithValue
import rlkits.utils as U
from rlkits.utils import colorize
import rlkits.utils.logger as logger
from rlkits.utils.math import explained_variance

import torch 
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_


@contextmanager
def timed(msg):
    print(colorize(msg, color='magenta'))
    tstart = time.time()
    yield
    print(colorize("done in %.3f seconds"%(time.time() - tstart), color='magenta'))


def sf01(arr):
    """
    aggregate experiences from all envs 
    each expr from one env can be used for one update
    I want to expr from the same env to stick together
    This means I need to tranpose the array so that
    (nenvs, nsteps, ...)
    so that when I reshape (C style) the array to merge the first two axes
    the exprs from the same env are contiguous     
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


def compute_loss(pi, trajectory, midx, eps, log_dir):
    """compute loss for policy and value net"""
    # policy loss
    old_log_prob = trajectory['log_prob'][midx]
    old_log_prob = torch.from_numpy(old_log_prob)
    
    adv = trajectory['adv'][midx]
    #adv = (adv - adv.mean()) / adv.std()
    adv = torch.from_numpy(adv)

    obs = trajectory['obs'][midx]
    obs = torch.from_numpy(obs)
    
    dist = pi.dist(pi.policy_net(obs))
    if dist is None:
        logger.log('Got NaN -- Bad')
        pi.save_ckpt()
        args = {
            "trajectory": trajectory,
            "midx": midx,
            "eps": eps
        }
        with open(os.path.join(ckpt_dir, 'local.pkl'), 'wb') as f:
            pickle.dump(args, f)
        sys.exit()
        
        
    
    log_prob = dist.log_prob(dist.sample())

    # importance sampling ratio
    ratio = torch.exp(log_prob - old_log_prob)
    if len(ratio.shape) > 1:
        ratio = ratio.squeeze(dim=1)
    assert ratio.shape == adv.shape, f"ratio shape: {ratio.shape}, adv shape : {adv.shape}"
    
    # if A(s, a) > 0, then maximize pi/pi_k, subject to pi/pi_k < 1 + \eps
    # this is equivalent to maximize 
    # (pi/pi_k) A
    # subject to (pi/pi_k)A < (1 + \eps)A
    
    # if A(s, a) < 0, then minimize pi/pi_k, subject to pi/pi_k > 1-\eps
    # this is equivalent to maximize pi/pi_k A, subject to 
    # (pi/pi_k)A < (1-\eps)A
    
    upper = torch.zeros_like(ratio, dtype=torch.float32).fill_(1 + eps)
    lower = torch.zeros_like(ratio, dtype=torch.float32).fill_(1 - eps)
    
    pm = (adv > 0) # positive advantage mask
    pos_loss = torch.maximum((ratio * adv)[pm], (upper*adv)[pm])
    
    
    nm = (adv < 0) # negative advantage mask
    neg_loss = torch.maximum((ratio*adv)[nm], (lower*adv)[nm])
    
    
    # want to maximize pos_loss and neg_loss
    # via backprop
    # i.e. objective should be to minimize -(pos_loss + neg_loss)
    pi_loss = -0.5*(pos_loss.mean() + neg_loss.mean()) 
    
    
    # value loss
    # want to make sure the difference between new value prediction
    # and the new value prediction is clipped within [-\eps, eps]
    
    
    vpreds = pi.value_net(obs)
    mQ = trajectory['Q'][midx]
    mQ = torch.from_numpy(mQ)
    
    if len(vpreds.shape) > 1:
        vpreds = vpreds.squeeze(dim=1)
    assert vpreds.shape == mQ.shape, f"vpreds shape: {vpreds.shape}, mQ shape : {mQ.shape}"

    v_loss = F.mse_loss(vpreds, mQ)
    return pi_loss + 0.1*v_loss.mean()


def PPO_clip(*,
    env,
    nsteps,
    total_timesteps,
    eps,            # epsilon
    pi_lr,          # policy learning rate
    v_lr,           # value net learning rate
    epochs,         # number of training epochs per policy update
    batch_size,
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

    pi = PolicyWithValue(ob_space=ob_space,
        ac_space=ac_space, ckpt_dir=ckpt_dir,
        **network_kwargs)
    poptimizer = optim.Adam(pi.policy_net.parameters(),
        lr=pi_lr)
    voptimizer = optim.Adam(pi.value_net.parameters(),
        lr=v_lr)

    sampler = ParallelEnvTrajectorySampler(env, pi, nsteps, 
        reward_transform=reward_transform) # hard-coded for pendulum
    
    rolling_buf_episode_rets = deque(maxlen=100) # moving average of last 10 episode returns
    rolling_buf_episode_lens = deque(maxlen=100)  # moving average of last 10 episode length
    
    
    # clip range  should decay from \eps to \eps * 1/n linearly
    # n is the total number of updates
    # this means at the begining, we want the agent to explore
    # more and we want the policy to converge at the end
    
    nframes = env.nenvs * nsteps # number of frames processed by update iter
    nupdates = total_timesteps // nframes
    
    
    for update in range(1, nupdates+1):
        tstart = time.perf_counter()
        
        trajectory = sampler(callback=estimate_Q)
        
        # aggregate exps from parallel envs
        for k, v in trajectory.items():
            if isinstance(v, np.ndarray):
                trajectory[k] = sf01(v)
        
        adv = trajectory['Q'] - trajectory['vpreds']
        trajectory['adv'] = (adv - adv.mean())/adv.std()
        
        piw, vw = pi.average_weight()
        logger.record_tabular('PolicyNetAvgW', piw.numpy())
        logger.record_tabular('ValueNetAvgW', vw.numpy())
        
        vqdiff = np.mean((trajectory['Q'] - trajectory['vpreds'])**2)
        logger.record_tabular('VQDiff', vqdiff)
        logger.record_tabular('Q', np.mean(trajectory['Q']))
        logger.record_tabular('vpreds', np.mean(trajectory['vpreds']))
        
        # determine the clip range for the current update step
        frac = 1.0 - (update - 1.0)/nupdates
        cliprange = 0.5*eps * frac + 0.5*eps

        # update policy
        idx = np.arange(len(trajectory['obs']))
        lossvals = []
        for _ in range(epochs):
            np.random.shuffle(idx) 
            for i in range(0, len(idx), batch_size):
                midx = idx[i:i+batch_size] # indices of exps to train
                loss = compute_loss(pi, trajectory, midx, cliprange, log_dir) 

                poptimizer.zero_grad()
                voptimizer.zero_grad()
                loss.backward()

                clip_grad_norm_(
                    pi.policy_net.parameters(),
                    max_norm=max_grad_norm)
                clip_grad_norm_(
                    pi.value_net.parameters(),
                    max_norm=max_grad_norm)

                poptimizer.step()
                voptimizer.step()
                lossvals.append(loss.detach().numpy())
        
        tnow = time.perf_counter()
        fps = int(nframes / (tnow - tstart)) # frames per seconds
        
        # logging
        if update % log_interval == 0 or update==1:
            
            logger.record_tabular('iteration/nupdates', f"{update}/{nupdates}")
            logger.record_tabular('cliprange', cliprange)
            
            for ep_rets in trajectory['ep_rets']:
                rolling_buf_episode_rets.extend(ep_rets)

            for ep_lens in trajectory['ep_lens']:
                rolling_buf_episode_lens.extend(ep_lens)

            
            # explained variance
            ev = explained_variance(trajectory['vpreds'], trajectory['Q'])
            logger.record_tabular('explained_variance', ev)


            piw, vw = pi.average_weight()
            logger.record_tabular('PolicyNetAvgW', piw.numpy())
            logger.record_tabular('ValueNetAvgW', vw.numpy())

            logger.record_tabular('loss', np.mean(lossvals))

            vqdiff = np.mean((trajectory['Q'] - trajectory['vpreds'])**2)
            
            logger.record_tabular('VQDiff', vqdiff)
            logger.record_tabular('Q', np.mean(trajectory['Q']))
            logger.record_tabular('vpreds', np.mean(trajectory['vpreds']))
            
            logger.record_tabular('FPS', fps)
            logger.record_tabular("MAEpRet", safemean(rolling_buf_episode_rets))
            logger.record_tabular('MAEpLen', safemean(rolling_buf_episode_lens))
            logger.record_tabular('MeanStepRew', safemean(trajectory['rews']))

            logger.dump_tabular()

            pi.save_ckpt()
            torch.save(poptimizer, os.path.join(ckpt_dir, 'optim.pth'))
            torch.save(voptimizer, os.path.join(ckpt_dir, 'optim.pth'))
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
    
    env=ParallelEnvBatch(make_env, nenvs=4)
    
    PPO_clip(
        env=env,
        nsteps=32,
        total_timesteps=4*32*1000, 
        eps = 0.2,
        pi_lr=1e-4,
        v_lr = 1e-4,
        epochs=1,
        batch_size=64, 
        log_interval=10,
        max_grad_norm=0.1,
        reward_transform=None,
        log_dir='/home/ubuntu/tmp/logs/ppo/cartpole',
        ckpt_dir='/home/ubuntu/tmp/models/ppo/cartpole',
        hidden_layers=[1024],
        activation=torch.nn.Tanh, 
        )
    
    env.close()



