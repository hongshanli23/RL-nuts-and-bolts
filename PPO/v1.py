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

from rlkits.sampler import TrajectorySampler
from rlkits.sampler import estimate_Q
from rlkits.policies import PolicyWithValue
import rlkits.utils as U
from rlkits.utils import colorize
import rlkits.utils.logger as logger

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


def compute_loss(pi, trajectory, midx, eps, log_dir):
    """compute loss for policy and value net"""
    # policy loss
    old_log_prob = trajectory['log_prob'][midx]
    old_log_prob = torch.from_numpy(old_log_prob)
    
    adv = trajectory['adv'][midx]
    adv = (adv - adv.mean()) / adv.std()
    adv = torch.from_numpy(adv)

    obs = trajectory['obs'][midx]
    obs = torch.from_numpy(obs)
    
    dist = pi.dist(pi.policy_net(obs))
    if dist is None:
        print('Got NaN -- Bad')
        pi.save_ckpt(log_dir)
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
    

    isg = ratio * adv    
    pi_loss1 = -isg
    upper = torch.zeros_like(isg, dtype=torch.float32).fill_(1 + eps)
    lower = torch.zeros_like(isg, dtype=torch.float32).fill_(1 - eps)

    pi_loss2 = -torch.maximum(
        torch.minimum(ratio, upper),  lower)*adv

    pi_loss = torch.maximum(pi_loss1, pi_loss2).mean() 
    # value loss
    vpreds = pi.value_net(obs)
    mQ = trajectory['Q'][midx]
    mQ = torch.from_numpy(mQ)
    
    if len(vpreds.shape) > 1:
        vpreds = vpreds.squeeze(dim=1)
    assert vpreds.shape == mQ.shape, f"vpreds shape: {vpreds.shape}, mQ shape : {mQ.shape}"

    v_loss = F.mse_loss(vpreds, mQ)
    return pi_loss.mean() + v_loss.mean()


def PPO_clip(*,
    env,
    nsteps,
    total_timesteps,
    eps,            # epsilon
    pi_lr,          # policy learning rate
    v_lr,           # value net learning rate
    epochs,         # number of training epochs per policy update
    batch_size,  
    log_dir,
    ckpt_dir,
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
        hidden_layers=[32, 32, 32],
        activation=torch.nn.Tanh)

    poptimizer = optim.Adam(pi.policy_net.parameters(),
        lr=pi_lr)
    voptimizer = optim.Adam(pi.value_net.parameters(),
        lr=v_lr)

    sampler = TrajectorySampler(env, pi, nsteps, 
        reward_transform=lambda x: (x + 8)/16) # hard-coded for pendulum
    
    rolling_buf_episode_rets = deque(maxlen=10) # moving average of last 10 episode returns
    rolling_buf_episode_lens = deque(maxlen=10)  # moving average of last 10 episode length
    while sampler.total_timesteps < total_timesteps:
        trajectory = sampler(callback=estimate_Q)
        trajectory['adv'] = trajectory['Q'] - trajectory['vpreds']
        
        piw, vw = pi.average_weight()
        logger.record_tabular('PolicyNetAvgW', piw.numpy())
        logger.record_tabular('ValueNetAvgW', vw.numpy())
        
        vqdiff = np.mean((trajectory['Q'] - trajectory['vpreds'])**2)
        logger.record_tabular('VQDiff', vqdiff)
        logger.record_tabular('Q', np.mean(trajectory['Q']))
        logger.record_tabular('vpreds', np.mean(trajectory['vpreds']))

        # update policy
        idx = np.arange(len(trajectory['obs']))
        lossvals = []
        for _ in range(epochs):
            np.random.shuffle(idx) 
            for i in range(0, len(idx), batch_size):
                midx = idx[i:i+batch_size] # indices of exps to train
                loss = compute_loss(pi, trajectory, midx, eps, log_dir) 

                poptimizer.zero_grad()
                voptimizer.zero_grad()
                loss.backward()

                clip_grad_norm_(
                    pi.policy_net.parameters(),
                    max_norm=0.1)
                clip_grad_norm_(
                    pi.value_net.parameters(),
                    max_norm=0.1)

                poptimizer.step()
                voptimizer.step()
                lossvals.append(loss.detach().numpy())
        
        logger.record_tabular('loss', np.mean(lossvals))
        
        # more logging
        rolling_buf_episode_rets.extend(trajectory['ep_rets'])
        rolling_buf_episode_lens.extend(trajectory['ep_lens'])
        
        logger.record_tabular("MAEpRet", np.mean(rolling_buf_episode_rets))
        logger.record_tabular('MAEpLen', np.mean(rolling_buf_episode_lens))
        logger.record_tabular('MeanStepRew', np.mean(trajectory['rews']))
        
        logger.dump_tabular()

    pi.save_ckpt()
    torch.save(poptimizer, os.path.join(ckpt_dir, 'optim.pth'))
    torch.save(voptimizer, os.path.join(ckpt_dir, 'optim.pth'))
    return

if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    PPO_clip(
        env=env,
        nsteps=1024,
        total_timesteps=1024*10000,
        eps = 0.1,
        pi_lr=1e-4,
        v_lr = 1e-4,
        epochs=4,
        batch_size=256, 
        log_dir='/home/ubuntu/tmp/logs/ppo/pendulum',
        ckpt_dir='/home/ubuntu/tmp/models/ppo/pendulum',
        )



