# When computing loss of a pi, should we use the action from the
# replay buffer (older pi), or should we use the action sampled
# from the current pi ?

# According to SAC paper, it seems that value functions are updated
# using actions sampled from the current pi, whereas the pi is
# updated through the actions sampled from the current pi via
# reparametrization trick. Why it makes sense to do it this way?

# For value net update, using the action sampled from the current
# pi amounts to a better label. The current pi makes an
# action so that the corresponding state action value Q(s, a) is a more
# accurate approximation of the state value V(s).

# For pi update, we need the pi gradient of the current
# pi, so naturally we should sample an action (for a state
# retrieved from replay buffer).

# In DDPG, the same better label for value function is given by
# the state action approximation of the target net over action
# sampled from target pi (on the next state).

# the reparametrization makes the distribution of
# acs_curr independent from the parameters of the
# pi net.
# reparametrization is a way to write the expectation
# independent from the parameter

# The advantage of reparametrization
# https://gregorygundersen.com/blog/2018/04/29/reparameterization/
# 1. It allows us to re-write gradient of expectation
# as expectation of gradient. Hence, we can use Monte
# Carlo method to estimate the gradient.
# 2. Stability: reparametrization limits the variance
# of the estimate. It basically caps the variance of
# the estimate to the variance of N(0,1)
# To see how reparam helps stability, checkout
# https://nbviewer.jupyter.org/github/gokererdogan/Notebooks/blob/master/Reparameterization%20Trick.ipynb

import os
from collections import deque
import gym
import numpy as np
from rlkits.memory import Memory
from rlkits.utils import to_tensor
from rlkits.policies import QNetForContinuousAction
from rlkits.policies import SACPolicy
from rlkits.env_wrappers import StartWithRandomActions,RecoverAction
import rlkits.utils.logger as logger

from copy import deepcopy
import torch
import torch.nn.functional as F
import torch.optim as optim


def compute_loss(pi, q1, q1_targ, q2, q2_targ,
                 batch, gamma, alpha):
    obs, acs, rews, nxs, dones = batch['obs0'], batch['actions'],\
        batch['rewards'], batch['obs1'], batch['terminals1']

    obs, acs, rews, nxs, dones = to_tensor(
        obs, acs, rews, nxs, dones)
    
    # compute the target for the Q-nets
    # treat nxa and nxa_logprob as constant
    with torch.no_grad():
        nxa, nxa_logprob = pi(nxs)
        nxv = torch.minimum(
            q1_targ(nxs, nxa.detach()), q2_targ(nxs, nxa.detach())
        )
    
    assert rews.shape == nxv.shape, f"{rews.shape}, {nxv.shape}"    
    y = rews + gamma*(1-dones)*(nxv - alpha*nxa_logprob.detach())

    # compute value loss
    q1, q2 = q1(obs, acs), q2(obs, acs)
    q1_loss = F.mse_loss(q1, y)
    q2_loss = F.mse_loss(q2, y)
    
    # pytorch support multiple forward pass
    # all computation graphs are saved
    # https://discuss.pytorch.org/t/multiple-forward-passes-single-conditional-backward-pass/99277
    # this means I can compute loss f
    
    # compute pi loss
    # objective is to maximize 
    # Q(s, a) - \alpha \log \pi(a | s), a sampled from the current pi
    acs_t, acs_t_logprob = pi.step(obs, no_grad=False)
    pi_loss = -(torch.minimum(q1(obs, acs_t), q2(obs, acs_t)) - alpha*acs_t_logprob)
    
    # @TODO
    # instrument entropy of the squashed gaussian dist
    # derive the formula for it based on Jacobian formula
         
    res = {
            "pi_loss": pi_loss.mean(),
            "q1_loss": q1_loss.mean(),
            "q2_loss": q2_loss.mean(),
          }
    return res
    


def SAC(*,
        env_name,
        nsteps,
        buf_size,
        warm_up_steps,
        gamma,
        alpha,
        v_lr,
        pi_lr,
        polyak,
        batch_size,
        log_interval,
        ckpt_dir,
        **network_kwargs
        ):
    
    # env
    def make_env():
        env = gym.make(env_name)
        env = StartWithRandomActions(env, max_random_actions=5)
        # env = RecoverAction(env)
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
    
    # Q-net
    q1 = QNetForContinuousAction(
        ob_space=ob_space, ac_space=ac_space,
        ckpt_dir=ckpt_dir,
        **network_kwargs
        )
    q2 = deepcopy(q1)
    
    q1_optim = optim.Adam(q1.model.parameters(), lr=v_lr)
    q2_optim = optim.Adam(q2.model.parameters(), lr=v_lr)
    
    # target Q-net
    q1_targ = deepcopy(q1)
    q2_targ = deepcopy(q1)
    
    # pi 
    pi = SACPolicy(
        ob_space=ob_space, ac_space=ac_space, ckpt_dir=ckpt_dir,
        **network_kwargs
    )
    pi_optim = optim.Adam(pi.model.parameters(), lr=pi_lr)    

    # memory buffer
    replay_buffer = Memory(
        limit=buf_size,
        action_shape=ac_space.shape
    )

    
    best_ret = np.float('-inf')
    rolling_buf_episode_rets = deque(maxlen=100)
    curr_state = env.reset()

    step=0
    while step <= nsteps:
        if step < warm_up_steps:
            action = pi.random_action()
        else:
            action, _ = pi.step(curr_state, no_grad=True)
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
            pi.reset()  # reset random process
            rolling_buf_episode_rets.append(episode_rews)
            episode_rews = 0
        else:
            curr_state = nx

        # train after warm up steps
        if step < warm_up_steps: continue
        batch = replay_buffer.sample(batch_size)
        losses = compute_loss(pi, q1, q2, batch,
                              gamma, alpha)
        
        pi_loss, q1_loss, q2_loss = losses['pi_loss'], \
            losses['q2_loss'], losses['q2_loss']

        pi_optim.zero_grad()
        pi_loss.backward()
        pi_optim.step()
        
        q1_optim.zero_grad()
        q1_loss.backward()
        q1_optim.step()

        q2_optim.zero_grad()
        q2_loss.backward()
        q2_optim.step()

        # update target Q-nets
        for p, p_targ in zip(q1.parameters(), q1_targ.parameters()):
            p_targ.data.copy_(polyak*p_targ.data + (1-polyak)*p.data)

        for p, p_targ in zip(q2.parameters(), q2_targ.parameters()):
            p_targ.data.copy_(polyak*p_targ.data + (1-polyak)*p.data)
        
        if step % log_interval == 0 and step > warm_up_steps:
            # loss from the policy and value net
            for k, v in losses.items():
                logger.record_tabular(k, np.mean(v.detach().numpy())) 
            
            # episodic returns
            ret = np.mean(rolling_buf_episode_rets)
            logger.record_tabular('ma_ep_ret', ret)
        

            
        
         
        
