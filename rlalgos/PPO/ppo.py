# PPO 
from collections import deque, defaultdict
import numpy as np
import time
import os
import gym
import pprint as pp
import pickle
import sys

from rlkits.sampler import ParallelEnvTrajectorySampler
from rlkits.sampler import estimate_Q, aggregate_experience
from rlkits.policies import PolicyWithValue
import rlkits.utils as U
import rlkits.utils.logger as logger
from rlkits.utils.math import explained_variance, KL
from rlkits.utils.context import timed


import torch 
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_


def compute_loss(pi, trajectory, midx, eps, ent_coef, log_dir):
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
        
    # DEADLY BUG
    #log_prob = dist.log_prob(dist.sample()) 
    
    actions = torch.from_numpy(trajectory['actions'][midx])
    log_prob = dist.log_prob(actions)
    
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
    return pi_loss + v_loss.mean() - ent_coef * dist.entropy().mean()



# TODO : merge with above loss fn
def compute_loss_kl(oldpi, pi, trajectory, midx, eps, log_dir):
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
            
    actions = torch.from_numpy(trajectory['actions'][midx])
    log_prob = dist.log_prob(actions)
    
    # importance sampling ratio
    ratio = torch.exp(log_prob - old_log_prob)
    if len(ratio.shape) > 1:
        ratio = ratio.squeeze(dim=1)
    assert ratio.shape == adv.shape, f"ratio shape: {ratio.shape}, adv shape : {adv.shape}"
    
    # Compute KL
    with torch.no_grad():
        oldpi_dist = oldpi.dist(oldpi.policy_net(obs))
    
    kl = KL(oldpi_dist, dist).mean()
    
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
    
    losses = {
        "surr_gain": (ratio*adv).mean(),
        "meankl": kl,
        "entropy": dist.entropy().mean(),
        "v_loss": v_loss   
    }
    return losses


def sync_policies(oldpi, pi):
    # oldpi <- pi
    oldpi.policy_net.load_state_dict(pi.policy_net.state_dict())
    oldpi.value_net.load_state_dict(pi.value_net.state_dict())
    return



def PPO(*,
    env,
    nsteps,
    total_timesteps,
    max_kl,
    beta,
    eps,            # epsilon
    gamma,          # discount factor
    pi_lr,          # policy learning rate
    v_lr,           # value net learning rate
    ent_coef,
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
    
    oldpi = PolicyWithValue(ob_space=ob_space,
        ac_space=ac_space, ckpt_dir=ckpt_dir,
        **network_kwargs)
    
    poptimizer = optim.Adam(pi.policy_net.parameters(),
        lr=pi_lr)
    voptimizer = optim.Adam(pi.value_net.parameters(),
        lr=v_lr)

    sampler = ParallelEnvTrajectorySampler(env, oldpi, nsteps, 
        reward_transform=reward_transform, gamma=gamma) #
    
    rolling_buf_episode_rets = deque(maxlen=100) 
    rolling_buf_episode_lens = deque(maxlen=100) 
    
    nframes = env.nenvs * nsteps # number of frames processed by update iter
    nupdates = total_timesteps // nframes
    
    best_ret = np.float('-inf')
    
    start = time.perf_counter()
    
    for update in range(1, nupdates+1):
        sync_policies(oldpi, pi)
        
        tstart = time.perf_counter()

        trajectory = sampler(callback=estimate_Q)
        
        # aggregate exps from parallel envs
        for k, v in trajectory.items():
            if isinstance(v, np.ndarray):
                trajectory[k] = aggregate_experience(v)
        
        adv = trajectory['Q'] - trajectory['vpreds']
        trajectory['adv'] = (adv - adv.mean())/adv.std()
        
        
        # determine the clip range for the current update step
        frac = 1.0 - (update - 1.0)/nupdates
        cliprange = 0.5*eps * frac + 0.5*eps

        # update policy
        idx = np.arange(len(trajectory['obs']))
        lossvals = defaultdict(list)
        for _ in range(epochs):
            np.random.shuffle(idx) 
            for i in range(0, len(idx), batch_size):
                midx = idx[i:i+batch_size] # indices of exps to train
                
                losses = compute_loss_kl(
                    oldpi=oldpi,
                    pi=pi, 
                    trajectory=trajectory, 
                    midx=midx, 
                    eps=cliprange, 
                    log_dir=log_dir
                ) 
                
                meankl = losses['meankl'].detach().item()
                if meankl > 1.5 * max_kl:
                    beta *= 2
                elif meankl < max_kl / 1.5:
                    beta /=2
                
                p_loss = -(losses['surr_gain'] - beta * losses['meankl'] + \
                           frac*ent_coef*losses['entropy'])
                poptimizer.zero_grad()
                p_loss.backward()
                clip_grad_norm_(
                    pi.policy_net.parameters(), max_norm=max_grad_norm
                )
                poptimizer.step()
                
                v_loss = losses['v_loss']
                voptimizer.zero_grad()
                v_loss.backward()
                clip_grad_norm_(
                    pi.value_net.parameters(), max_norm=max_grad_norm
                )
                voptimizer.step()
                
                for k, v in losses.items():
                    lossvals[k].append(v.detach().item())
        
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
            logger.record_tabular('policy_net_weight', piw.numpy())
            logger.record_tabular('value_net_weight', vw.numpy())

            # losses 
            for k, v in lossvals.items():
                logger.record_tabular(k, np.mean(v))
            
            vqdiff = np.mean((trajectory['Q'] - trajectory['vpreds'])**2)
            
            logger.record_tabular('VQDiff', vqdiff)
            logger.record_tabular('Q', np.mean(trajectory['Q']))
            logger.record_tabular('vpreds', np.mean(trajectory['vpreds']))
            
            logger.record_tabular('FPS', fps)
            
            ret = safemean(rolling_buf_episode_rets)
            logger.record_tabular("ma_ep_ret", ret)
            logger.record_tabular('ma_ep_len', safemean(rolling_buf_episode_lens))
            logger.record_tabular('mean_step_rew', safemean(trajectory['rews']))

            logger.dump_tabular()

            if ret !=np.nan and ret > best_ret:
                best_ret = ret
                pi.save_ckpt('best')
                torch.save(poptimizer, os.path.join(ckpt_dir, 'poptim-best.pth'))
                torch.save(voptimizer, os.path.join(ckpt_dir, 'voptim-best.pth'))
    
    end = time.perf_counter()
    logger.log(f'Total training time: {end - start}')
    
    pi.save_ckpt('final')
    torch.save(poptimizer, os.path.join(ckpt_dir, 'poptim-final.pth'))
    torch.save(voptimizer, os.path.join(ckpt_dir, 'voptim-final.pth'))
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
    
    def pendulum():
        env = gym.make('Pendulum-v0')
        env = AutoReset(env)
        env = StartWithRandomActions(env, max_random_actions=5)
        return env
    
    nenvs = 16
    nsteps = 128
    env=ParallelEnvBatch(make_env, nenvs=nenvs)
    PPO(
        env=env,
        nsteps=nsteps,
        total_timesteps=nenvs*nsteps*10000, 
        max_kl=1e-2,
        beta=0.5,
        eps = 0.2,
        gamma=0.99,
        pi_lr=1e-4,
        v_lr = 1e-4,
        ent_coef=0.0,
        epochs=3,
        batch_size=nenvs*nsteps, 
        log_interval=10,
        max_grad_norm=0.1,
        reward_transform=None,
        log_dir='/home/ubuntu/reinforcement-learning/experiments/ppo/pendulum/0',
        ckpt_dir='/home/ubuntu/reinforcement-learning/experiments/ppo/pendulum/0',
        hidden_layers=[256, 256, 64],
        activation=torch.nn.ReLU, 
        )
    
    env.close()



