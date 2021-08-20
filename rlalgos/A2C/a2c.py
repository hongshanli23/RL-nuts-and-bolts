# A2C

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
from rlkits.sampler import shuffle_experience
from rlkits.policies import PolicyWithValue
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
    dist = pi.dist(pi.policy_net(obs))

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

    vpreds = pi.value_net(obs)
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


def sync_policies(oldpi, pi):
    # oldpi <- pi
    oldpi.policy_net.load_state_dict(pi.policy_net.state_dict())
    oldpi.value_net.load_state_dict(pi.value_net.state_dict())
    return


def policy_diff(oldpi, pi):
    """Compute the average distance between params of oldpi and pi"""
    diff = 0.0
    cnt = 0
    for p1, p2 in zip(oldpi.policy_net.parameters(), pi.policy_net.parameters()):
        diff += torch.mean(torch.abs(p1.data - p2.data))
        cnt +=1
    return diff / cnt


def A2C(
    env,
    nsteps,
    gamma,
    total_timesteps,
    pi_lr,
    v_lr,
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

    pi = PolicyWithValue(ob_space=ob_space,
        ac_space=ac_space, ckpt_dir=ckpt_dir,
        **network_kwargs)

    # only used to compute policy difference
    oldpi = PolicyWithValue(ob_space=ob_space,
        ac_space=ac_space, ckpt_dir=ckpt_dir,
        **network_kwargs)

    poptimizer = optim.Adam(pi.policy_net.parameters(),
        lr=pi_lr)
    voptimizer = optim.Adam(pi.value_net.parameters(),
        lr=v_lr)

    sampler = ParallelEnvTrajectorySampler(env, pi, nsteps,
        reward_transform=reward_transform, gamma=gamma)

    # moving average of last 10 episode returns
    rolling_buf_episode_rets = deque(maxlen=10)

    # moving average of last 10 episode length
    rolling_buf_episode_lens = deque(maxlen=10)


    nframes = env.nenvs * nsteps # number of frames processed by update iter
    nupdates = total_timesteps // nframes

    start = time.perf_counter()
    best_ret = np.float('-inf')
    for update in range(1, nupdates+1):
        sync_policies(oldpi, pi)

        tstart = time.perf_counter()

        trajectory = sampler(callback=estimate_Q)

        # aggregate exps from parallel envs
        for k, v in trajectory.items():
            if isinstance(v, np.ndarray):
                trajectory[k] = aggregate_experience(v)

        #trajectory = shuffle_experience(trajectory)

        adv = trajectory['Q'] - trajectory['vpreds']
        trajectory['adv'] = (adv - adv.mean())/adv.std()

        losses = compute_loss(pi=pi,
                           trajectory=trajectory,
                           log_dir=log_dir)

        frac = 1.0 - (update - 1.0)/nupdates
        loss = losses['pi_loss'] + losses['v_loss'] \
            - ent_coef * frac * losses['entropy']


        poptimizer.zero_grad()
        voptimizer.zero_grad()
        loss.backward()

        clip_grad_norm_(pi.policy_net.parameters(),
                       max_norm=max_grad_norm)
        clip_grad_norm_(pi.value_net.parameters(),
                        max_norm=max_grad_norm)

        poptimizer.step()
        voptimizer.step()

        tnow = time.perf_counter()


        # logging
        if update % log_interval == 0 or update==1:
            fps = int(nframes / (tnow - tstart)) # frames per seconds

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


            step_size = policy_diff(oldpi, pi)
            logger.record_tabular('step_size', step_size.numpy())

            # explained variance
            ev = explained_variance(trajectory['vpreds'], trajectory['Q'])
            logger.record_tabular('explained_variance', ev)


            piw, vw = pi.average_weight()
            logger.record_tabular('policy_net_weight', piw.numpy())
            logger.record_tabular('value_net_weight', vw.numpy())


            vqdiff = np.mean((trajectory['Q'] - trajectory['vpreds'])**2)

            logger.record_tabular('vqdiff', vqdiff)
            logger.record_tabular('Q', np.mean(trajectory['Q']))
            logger.record_tabular('vpreds', np.mean(trajectory['vpreds']))



            logger.record_tabular('FPS', fps)

            ret =  safemean(rolling_buf_episode_rets)

            logger.record_tabular("ma_ep_ret", ret)
            logger.record_tabular('ma_ep_len',
                                  safemean(rolling_buf_episode_lens))
            logger.record_tabular('mean_rew_step',
                                  np.mean(trajectory['rews']))


            if ret != np.nan and ret > best_ret:
                best_ret = ret
                pi.save_ckpt('best')


            logger.dump_tabular()

    pi.save_ckpt('last')
    torch.save(poptimizer, os.path.join(ckpt_dir, 'optim.pth'))
    torch.save(voptimizer, os.path.join(ckpt_dir, 'optim.pth'))
    return


def safemean(l):
    return np.nan if len(l) == 0 else np.mean(l)


if __name__ == '__main__':
    from rlkits.env_batch import ParallelEnvBatch
    from rlkits.env_wrappers import AutoReset, StartWithRandomActions
    from rlkits.env_wrappers import TransformReward, Truncate


    def stochastic_reward(rew):
        eps = np.random.normal(loc=0.0, scale=0.1, size=rew.shape)
        return rew + eps

    def make_env():
        env = gym.make('CartPole-v0').unwrapped
        env = AutoReset(env)
        env = StartWithRandomActions(env, max_random_actions=5)
        return env


    def normalize_pendulum(rew):
        """Reward normalizer for Pendulum"""
        return (rew + 8)

    def pendulum():
        """Make env for pendulum"""

        env = gym.make('Pendulum-v0')
        #env = TransformReward(env, normalize_pendulum)
        #env = Truncate(env, lower_bound=-10)
        env = AutoReset(env)
        return env

    nsteps=128
    nenvs = 8
    env=ParallelEnvBatch(pendulum, nenvs=nenvs)



    A2C(
        env=env,
        nsteps=nsteps,
        gamma=0.99,
        total_timesteps=nsteps*nenvs*10000,
        pi_lr=1e-4,
        v_lr=1e-4,
        ent_coef=0.0,
        log_interval=10,
        max_grad_norm=0.1,
        reward_transform=None,
        log_dir= '/home/ubuntu/reinforcement-learning/experiments/A2C_2/log/pendulum/5',
        ckpt_dir='/home/ubuntu/reinforcement-learning/experiments/A2C_2/log/pendulum/5',
        hidden_layers=[256, 256, 64],
        activation=torch.nn.ReLU
    )

