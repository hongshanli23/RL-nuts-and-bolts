# TRPO algorithm 

from rlkits.sampler import estimate_Q, aggregate_experience
from rlkits.sampler import ParallelEnvTrajectorySampler

from rlkits.policies import PolicyWithValue
from rlkits.utils.math import KL, conjugate_gradient
import rlkits.utils as U
from rlkits.utils import colorize
import rlkits.utils.logger as logger

import gym
import torch
import torch.optim as optim

from contextlib import contextmanager
from collections import deque
import numpy as np
import time
import os

@contextmanager
def timed(msg):
    print(colorize(msg, color='magenta'))
    tstart = time.time()
    yield
    print(colorize("done in %.3f seconds"%(time.time() - tstart), color='magenta'))

def compute_losses(oldpi, pi, trajectory):
    """Compute surrogate gain and policy gradient
    trajectory is sampled from the old policy;
    Use importance sampling to estimate the policy
    gain of pi relative to oldpi
    """

    obs = trajectory['obs']
    #print('obs dtype', obs.dtype)
    obs = torch.from_numpy(obs).float()

    actions = trajectory['actions']
    actions = torch.from_numpy(actions).float()

    pi_dist = pi.dist(pi.policy_net(obs))
    
    # no graph for old policy
    # it should be treated as a constant
    with torch.no_grad():
        oldpi_dist = oldpi.dist(oldpi.policy_net(obs))

    # estimate KL between oldpi and pi
    # should be 0 in this function call
    kl = KL(oldpi_dist, pi_dist).mean()

    # importance sampling ratio
    ratio = torch.exp(
        pi_dist.log_prob(actions) - oldpi_dist.log_prob(actions)
    )
    if len(ratio.shape) > 1:
        ratio = torch.squeeze(ratio, dim=1)

    # estimate advantage of the old policy
    adv = trajectory['Q'] - trajectory['vpreds']

    # normalize advantage
    adv = (adv - adv.mean())/adv.std()

    # estimate the surrogate gain
    adv = torch.from_numpy(adv)
    assert ratio.shape == adv.shape, f"ratio : {ratio.shape}, adv: {adv.shape}"
    surr_gain = torch.mean(ratio * adv)

    res = {
        'surr_gain': surr_gain,
        'meankl': kl,
        }
    return res

def compute_fvp(oldpi, pi, obs, p):
    """Compute Ap
    where A is the Hessian of KL(oldpi || pi)
    Use direct method to avoid explicitly computing A
    """
    obs = torch.from_numpy(obs)
    oldpi_dist = oldpi.dist(oldpi.policy_net(obs))
    pi_dist = pi.dist(pi.policy_net(obs))

    kl = KL(oldpi_dist, pi_dist).mean()
    klgrads = torch.autograd.grad(kl, 
        pi.policy_net.parameters(), create_graph=True)
    klgrads = U.flatten(klgrads)

    Ap = torch.autograd.grad(torch.dot(klgrads, p),
        pi.policy_net.parameters())
    return U.flatten(Ap)

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
        
def pendulum_reward_transform(rew):
    """normalize to [-1, 1]"""
    return (rew + 8.0)/16.0 


def TRPO(*, 
        env, 
        nsteps, 
        total_timesteps, 
        log_interval,
        log_dir, 
        ckpt_dir,
        reward_transform,
        gamma=0.99,
        max_kl=1e-2,
        cg_iters=10, 
        cg_damping=1e-2, 
        backtrack_steps=10, 
        v_iters=3,
        batch_size=64, 
        v_lr=1e-4,
        **network_kwargs):
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    logger.configure(dir=log_dir)

    ob_space = env.observation_space
    ac_space = env.action_space

    pi = PolicyWithValue(ob_space=ob_space, 
        ac_space=ac_space, ckpt_dir=ckpt_dir, **network_kwargs)
    
    
    oldpi = PolicyWithValue(ob_space=ob_space, 
        ac_space=ac_space, ckpt_dir=ckpt_dir, **network_kwargs)

    
    # optimizer for the value net
    voptimizer = optim.Adam(pi.value_net.parameters(),
        lr=v_lr)

    # env sampler
    sampler = ParallelEnvTrajectorySampler(
        env=env, 
        policy=pi,
        nsteps=nsteps, 
        reward_transform=reward_transform,
        gamma=gamma
    )
    
    def fisher_vector_product(p):
        """Fisher vector product
        Used for conjugate gradient algorithm
        """
        return compute_fvp(oldpi, pi, trajectory['obs'], p) + cg_damping*p
    
    
    best_ret = np.float('-inf')
    
    rolling_buf_episode_rets = deque(maxlen=10) 
    rolling_buf_episode_lens = deque(maxlen=10)
    
    start = time.perf_counter()
    
    nframes = nsteps * env.nenvs
    nupdates = total_timesteps // (nframes)
    for update in range(1, nupdates + 1):
        tstart = time.perf_counter()
        # oldpi <- pi
        sync_policies(oldpi, pi)
        trajectory = sampler(callback=estimate_Q)
        
        # aggregate exps from parallel envs
        for k, v in trajectory.items():
            if isinstance(v, np.ndarray):
                trajectory[k] = aggregate_experience(v)
        
        # losses before update (has gradient)
        lossesbefore = compute_losses(oldpi, pi, trajectory)
        
        # estimate policy gradient of pi
        g = torch.autograd.grad(lossesbefore['surr_gain'], 
            pi.policy_net.parameters())
        g = U.flatten(g)
        
        if torch.allclose(g, torch.zeros_like(g)):
            logger.log("Got zero gradient, not updating")
            continue

        with timed('conjugate gradient'):
            npg, cginfo = conjugate_gradient(
                fisher_vector_product, g, 
                cg_iters=cg_iters, verbose=True)
        assert torch.isfinite(npg).all()

        # stepsize of the update
        shs = torch.dot(npg, compute_fvp(
                oldpi, pi, trajectory['obs'], npg)).detach()
        
        stepsize = torch.sqrt(2*max_kl/shs)
        
        # backtrack line search
        params0 = U.flatten(pi.policy_net.parameters())
        expected_improve = torch.dot(g, stepsize * npg) # first order appr of surrgate gain
        for _ in range(backtrack_steps):
            newparams = params0 + stepsize * npg
            U.set_from_flat(pi.policy_net, newparams)
            with torch.no_grad():
                losses = compute_losses(oldpi, pi, trajectory)
                
            improve = losses['surr_gain'] - lossesbefore['surr_gain']
            logger.log("Expected: %.3f Actual: %.3f"%(expected_improve, improve))
            
            if any(not torch.isfinite(v).all() for _, v in losses.items()):
                logger.log('Got infinite loss!')
            elif losses['meankl'] > 1.5 * max_kl:
                logger.log('Violated KL contraint')
            elif improve < 0.0:
                logger.log('Surrogate gain not improving')
            else:
                logger.log('Step size is OK')
                break
            stepsize *= 0.5
        else:
            logger.log('Canot find a good step size, resume to the old poliy')
            U.set_from_flat(pi.policy_net, params0)
                
        # update value net
        obs, Q = trajectory['obs'], trajectory['Q']
        obs, Q = torch.from_numpy(obs), torch.from_numpy(Q)
        for _ in range(v_iters):
            for i in range(0, len(obs), batch_size):
                x, y = obs[i:i+batch_size], Q[i:i+batch_size]
                vpreds = pi.value_net(x).squeeze(dim=1)
                vloss = torch.mean((vpreds - y)**2) 
                voptimizer.zero_grad()
                vloss.backward()
                voptimizer.step()
                
        tnow = time.perf_counter()
        
        # logging
        if update % log_interval == 0 or update == 1:
            fps = int(nframes // (tnow - tstart))
            logger.record_tabular('FPS', fps)
            
            # policy loss
            for k, v in lossesbefore.items():
                logger.record_tabular(k, np.mean(v.detach().numpy()))
            
            # conjugate gradient info
            for k, v in cginfo.items():
                logger.record_tabular(k, v)
            
            # weights
            piw, vw = pi.average_weight()
            logger.record_tabular('policy_net_weight', piw.numpy())
            logger.record_tabular('value_net_weight', vw.numpy())
            
            # step size as the change in policy params
            step_size = policy_diff(oldpi, pi)
            logger.record_tabular('step_size', step_size.numpy())

            vqdiff = np.mean((trajectory['Q'] - trajectory['vpreds'])**2)
            logger.record_tabular('VQDiff', vqdiff)
            logger.record_tabular('Q', np.mean(trajectory['Q']))
            logger.record_tabular('vpreds', np.mean(trajectory['vpreds']))

            # more logging
            for ep_rets in trajectory['ep_rets']:
                rolling_buf_episode_rets.extend(ep_rets)

            for ep_lens in trajectory['ep_lens']:
                rolling_buf_episode_lens.extend(ep_lens)

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
    
    
    now = time.perf_counter()
    
    logger.log(f'Total training time: {now - start}')
    pi.save_ckpt('last')
    torch.save(voptimizer, os.path.join(ckpt_dir, 'optim.pth'))
    return 

def safemean(l):
    return np.nan if len(l) == 0 else np.mean(l)


if __name__ == '__main__':
    from rlkits.env_batch import ParallelEnvBatch
    from rlkits.env_wrappers import AutoReset, StartWithRandomActions
    
    
    def stochastic_reward(rew):
        eps = np.random.normal(loc=0.0, scale=0.1, size=rew.shape)
        return rew + eps
    
    def make_env():
        env = gym.make('CartPole-v0').unwrapped
        env = AutoReset(env)
        env = StartWithRandomActions(env, max_random_actions=5)
        return env
    
    nenvs = 16
    env = ParallelEnvBatch(make_env, nenvs=nenvs)
    
    TRPO(
        env=env,
        nsteps=32, 
        total_timesteps=nenvs*32*10000,
        gamma=0.99,
        log_interval=10,
        reward_transform=None,
        log_dir='/home/ubuntu/reinforcement-learning/experiments/TRPO/8',
        ckpt_dir='/home/ubuntu/reinforcement-learning/experiments/TRPO/8',
        max_kl=1e-3,
        cg_iters=20,
        cg_damping=1e-2,
        backtrack_steps=10,
        v_iters=1,
        batch_size=nenvs*32,
        v_lr=1e-4,
        hidden_layers=[32, 32, 32],
        activation=torch.nn.ReLU
    )