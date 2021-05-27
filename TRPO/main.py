# TRPO algorithm 

from rlkits.sampler import estimate_Q
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

def pendulum_reward_transform(rew):
    """normalize to [-1, 1]"""
    return (rew + 8.0)/16.0 

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

def TRPO(*, 
        env, 
        nsteps, 
        total_timesteps, 
        log_dir, 
        ckpt_dir,
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
    sampler = ParallelEnvTrajectorySampler(env, pi, nsteps, pendulum_reward_transform)
    
    def fisher_vector_product(p):
        """Fisher vector product
        Used for conjugate gradient algorithm
        """
        return compute_fvp(oldpi, pi, trajectory['obs'], p) + cg_damping*p
    
    rolling_buf_episode_rets = deque(maxlen=10) # moving average of last 10 episode returns
    rolling_buf_episode_lens = deque(maxlen=10)  # moving average of last 10 episode length
    while sampler.total_timesteps < total_timesteps:
        # oldpi <- pi
        sync_policies(oldpi, pi)
        trajectory = sampler(callback=estimate_Q)
        
        # aggregate exps from parallel envs
        for k, v in trajectory.items():
            if isinstance(v, np.ndarray):
                trajectory[k] = sf01(v)
        
        # losses before update (has gradient)
        lossesbefore = compute_losses(oldpi, pi, trajectory)

        for k, v in lossesbefore.items():
            logger.record_tabular(k, np.mean(v.detach().numpy()))

        # estimate policy gradient of pi
        g = torch.autograd.grad(lossesbefore['surr_gain'], 
            pi.policy_net.parameters())
        g = U.flatten(g)
        
        if torch.allclose(g, torch.zeros_like(g)):
            logger.log("Got zero gradient, not updating")
            continue

        with timed('conjugate gradient'):
            npg = conjugate_gradient(
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
        

        vqdiff = np.mean((trajectory['Q'] - trajectory['vpreds'])**2)
        logger.record_tabular('VQDiff', vqdiff)
        logger.record_tabular('Q', np.mean(trajectory['Q']))
        logger.record_tabular('vpreds', np.mean(trajectory['vpreds']))
        
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
        
        # more logging
        
        for ep_rets in trajectory['ep_rets']:
            rolling_buf_episode_rets.extend(ep_rets)

        for ep_lens in trajectory['ep_lens']:
            rolling_buf_episode_lens.extend(ep_lens)

        
        logger.record_tabular("MAEpRet", safemean(rolling_buf_episode_rets))
        logger.record_tabular('MAEpLen', safemean(rolling_buf_episode_lens))
        logger.record_tabular('MeanStepRew', np.mean(trajectory['rews']))
        
        logger.dump_tabular()

    pi.save_ckpt()
    torch.save(voptimizer, os.path.join(ckpt_dir, 'optim.pth'))
    return 

def safemean(l):
    return np.nan if len(l) == 0 else np.mean(l)
