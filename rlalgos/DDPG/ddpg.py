# Use parallel env as a quicker way to sample exprs
# Not as a way to solve sample inefficiency problem
# as in many on-policy algorithms

from collections import deque, defaultdict
import time
import numpy as np
import gym
import os

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

import rlkits.utils.logger as logger
from rlkits.policies import DeterministicPolicy
from rlkits.policies import QNetForContinuousAction
from rlkits.memory import Memory
from rlkits.evaluate import evaluate_policy
from rlkits.env_batch import ParallelEnvBatch
from rlkits.env_wrappers import AutoReset, StartWithRandomActions
from rlkits.env_wrappers import RecoverAction
from rlkits.running_mean_std import RunningMeanStd

from ipdb import set_trace


def to_tensor(*args):
    new_args = []
    for arg in args:
        assert isinstance(arg, np.ndarray)
        if arg.dtype == np.float64:
            arg = arg.astype(np.float32)
        new_args.append(torch.from_numpy(arg))
    return new_args


def compute_loss(
        policy,
        target_policy,
        value_net,
        target_value_net,
        batch,
        gamma):
    """use a batch of experiences sampled from replay buffer to
    compute the loss of policy and value net

    batch: a batch of experiences sampled from replay buffer
    gamma: discount factor
    """
    obs, acs, rews, nxs, dones = batch['obs0'], batch['actions'],\
        batch['rewards'], batch['obs1'], batch['terminals1']

    obs, acs, rews, nxs, dones = to_tensor(
        obs, acs, rews, nxs, dones)
    # target for value net
    with torch.no_grad():
        nx_state_vals = target_value_net(nxs,
                                         target_policy(nxs))
    assert rews.shape == nx_state_vals.shape, f"{rews.shape}, {nx_state_vals.shape}"
    q_targ = rews + (1 - dones)*gamma*nx_state_vals

    # predicted q-value for the current state and action
    q_pred = value_net(obs, acs)
    value_loss = F.mse_loss(q_pred, q_targ)

    # policy loss
    policy_loss = -value_net(obs, policy(obs)).mean()

    res = {
            "policy_loss": policy_loss,
            "value_loss": value_loss
          }
    return res


def DDPG(*,
         env_name,
         nsteps,
         buf_size,
         warm_up_steps,
         gamma,
         pi_lr,
         v_lr,
         polyak,
         batch_size,
         log_intervals,
         max_grad_norm,
         l2_weight_decay,
         log_dir,
         ckpt_dir,
         **network_kwargs,
         ):
    """
    env: gym env (parallel)
    nsteps: number of steps to sample from the parallel env
        nstep * env.nenvs frames will be sampled

    ployak (float): linear interpolation coefficient for updating
        the target policy and value net from the current ones;
        Interpret it as the weight of the current target network

    buf_size: size of the replay buffer

    normalize_action: clip the action to [-1, 1]
    """
    # env
    def make_env():
        env = gym.make(env_name)
        env = StartWithRandomActions(env, max_random_actions=5)
        env = RecoverAction(env)
        return env

    env = make_env()

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    logger.configure(dir=log_dir)

    ob_space = env.observation_space
    ac_space = env.action_space
    print("======", f"action space shape {ac_space.shape}, \
          ob space shape: {ob_space.shape}", "======")
    policy = DeterministicPolicy(
        ob_space=ob_space, ac_space=ac_space,
        ckpt_dir=ckpt_dir,
        **network_kwargs
    )
    target_policy = DeterministicPolicy(
        ob_space=ob_space, ac_space=ac_space,
        ckpt_dir=ckpt_dir,
        **network_kwargs
    )
    target_policy.model.load_state_dict(policy.model.state_dict())

    value_net = QNetForContinuousAction(
        ob_space=ob_space, ac_space=ac_space, ckpt_dir=ckpt_dir,
        **network_kwargs
    )

    target_value_net = QNetForContinuousAction(
        ob_space=ob_space, ac_space=ac_space, ckpt_dir=ckpt_dir,
        **network_kwargs
    )

    target_value_net.model.load_state_dict(value_net.model.state_dict())

    poptimizer = optim.Adam(policy.parameters(), lr=pi_lr,
                            weight_decay=l2_weight_decay)
    voptimizer = optim.Adam(value_net.parameters(), lr=v_lr,
                            weight_decay=l2_weight_decay)

    replay_buffer = Memory(
        limit=buf_size,
        action_shape=ac_space.shape,
        observation_shape=ob_space.shape
    )

    best_ret = np.float('-inf')
    rolling_buf_episode_rets = deque(maxlen=10)
    curr_state = env.reset()
    policy.reset()

    step = 0
    episode_rews = 0.0
    start = time.perf_counter()
    while step <= nsteps:
        # warm up steps
        if step < warm_up_steps:
            action = policy.random_action()
        else:
            action = policy.step(curr_state)
        nx, rew, done, _ = env.step(action)
        # record to the replay buffer
        assert nx.shape == ob_space.shape, f"{nx.shape},{ob_space.shape}"
        assert action.shape == ac_space.shape, f"{action.shape},{ac_space.shape}"
        replay_buffer.append(
            obs0=curr_state, action=action, reward=rew, obs1=nx, terminal1=done
        )
        episode_rews += rew
        if done:
            curr_state = env.reset()
            policy.reset()  # reset random process
            rolling_buf_episode_rets.append(episode_rews)
            episode_rews = 0
        else:
            curr_state = nx

        if step > warm_up_steps:
            # update policy and value
            batch = replay_buffer.sample(batch_size)
            losses = compute_loss(
                policy, target_policy, value_net,
                target_value_net, batch, gamma
            )
            ploss = losses['policy_loss']
            poptimizer.zero_grad()
            ploss.backward()
            poptimizer.step()

            vloss = losses['value_loss']
            voptimizer.zero_grad()
            vloss.backward()
            voptimizer.step()

            # update target value net and policy
            for p, p_targ in zip(
                    policy.parameters(),
                    target_policy.parameters()):
                p_targ.data.copy_(polyak*p_targ.data + (1-polyak)*p.data)

            for p, p_targ in zip(
                    value_net.parameters(),
                    target_value_net.parameters()):
                p_targ.data.copy_(polyak*p_targ.data + (1-polyak)*p.data)

        if step % log_intervals == 0 and step > warm_up_steps:
            # loss from policy and value
            for k, v in losses.items():
                logger.record_tabular(k, np.mean(v.detach().numpy()))

            ret = np.mean(rolling_buf_episode_rets)
            logger.record_tabular("ma_ep_ret", ret)

            pw, tpw = policy.average_weight(), target_policy.average_weight()
            vw, tvw = value_net.average_weight(), target_value_net.average_weight()
            logger.record_tabular("policy_net_weight", pw)
            logger.record_tabular("target_policy_net_weight", tpw)
            logger.record_tabular("value_net_weight", vw)
            logger.record_tabular("target_value_net_weight", tvw)

            logger.dump_tabular()
            if ret > best_ret:
                best_ret = ret
                policy.save_ckpt('policy-best')
                value_net.save_ckpt('value-best')
                torch.save(poptimizer, os.path.join(ckpt_dir,
                                                    'poptim-best.pth'))
                torch.save(voptimizer, os.path.join(ckpt_dir,
                                                    'voptim-best.pth'))
        step += 1
    end = time.perf_counter()
    logger.log(f"Total time elapsed: {end - start}")
    policy.save_ckpt('policy-final')
    value_net.save_ckpt('value-final')
    torch.save(poptimizer, os.path.join(ckpt_dir, 'poptim-final.pth'))
    torch.save(voptimizer, os.path.join(ckpt_dir, 'voptim-final.pth'))
    return


if __name__ == '__main__':
    DDPG(
        env_name='Pendulum-v0',
        nsteps=int(2e5),
        buf_size=int(2e5),
        warm_up_steps=int(1e3),
        gamma=0.99,
        pi_lr=1e-4,
        v_lr=1e-4,
        l2_weight_decay=1e-4,
        polyak=0.99,
        batch_size=128,
        log_intervals=int(2e3),
        max_grad_norm=0.1,
        log_dir="/tmp/ddpg",
        ckpt_dir="/tmp/ddpg",
        hidden_layers=[256, 256, 64]
    )

    # 32 x 8 = 256 exprs sampled per iter
    # 256000
