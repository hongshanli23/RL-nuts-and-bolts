# PPO 

from contextlib import contextmanager
from collections import deque
import numpy as np
import time
import os
import gym

from rlkits.sampler import TrajectorySampler
from rlkits.policies import PolicyWithValue
import rlkits.utils as U
from rlkits.utils import colorize
import rlkits.utils.logger as logger

import torch 
import torch.optim as optim


@contextmanager
def timed(msg):
    print(colorize(msg, color='magenta'))
    tstart = time.time()
    yield
    print(colorize("done in %.3f seconds"%(time.time() - tstart), color='magenta'))


def importance_sampling_gain(oldpi, pi, trajectory):
    obs = torch.from_numpy(trajectory['obs'])
    actions = trajectory['actions']
    actions = torch.from_numpy(actions)

    pi_dist = pi.dist(pi.policy_net(obs))

    # no graph for old policy
    with torch.no_grad():
        oldpi_dist = oldpi.dist(oldpi.policy_net(obs))

    # importance sampling ratio
    ratio = torch.exp(
        pi_dist.log_prob(actions) - oldpi_dist.log_prob(actions)
        )
    if len(ratio.shape) > 1:
        ratio = torch.squeeze(ratio, dim=1)
        
    adv = trajectory['Q'] - trajectory['vpreds']
    # adv = (adv - adv.mean()) / adv.std()
    adv = torch.from_numpy(adv)
    assert ratio.shape == adv.shape, f"ratio : {ratio.shape}, adv: {adv.shape}"

    return torch.mean(ratio * adv)

def sync_policies(oldpi, pi):
    # oldpi <- pi
    oldpi.policy_net.load_state_dict(pi.policy_net.state_dict())
    oldpi.value_net.load_state_dict(pi.value_net.state_dict())
    return

def PPO_clip(*,
    env,
    nsteps,
    total_timesteps,
    eps,            # epsilon
    pi_lr,          # policy learning rate
    v_lr,           # value net learning rate
    backtrack_steps, # line search backtrack steps
    log_dir,
    ckpt_dir,
    v_iters,
    batch_size):

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    logger.configure(dir=log_dir) 

    ob_space = env.observation_space
    ac_space = env.action_space

    pi = PolicyWithValue(ob_space=ob_space,
        ac_space=ac_space, hidden_layers=[1024],
        activation=torch.nn.ReLU)

    oldpi = PolicyWithValue(ob_space=ob_space,
        ac_space=ac_space, hidden_layers=[1024],
        activation=torch.nn.ReLU)
    
    voptimizer = optim.Adam(pi.value_net.parameters(),
        lr=v_lr)

    sampler = TrajectorySampler(env, pi, nsteps, 
        reward_transform=lambda x: (x + 8)/16) # hard-coded for pendulum
    
    rolling_buf_episode_rets = deque(maxlen=10) # moving average of last 10 episode returns
    rolling_buf_episode_lens = deque(maxlen=10)  # moving average of last 10 episode length
    while sampler.total_timesteps < total_timesteps:
        sync_policies(oldpi, pi)

        trajectory = sampler()
        isg = importance_sampling_gain(oldpi, pi, trajectory)
        
        # policy gradient
        g = torch.autograd.grad(isg, pi.policy_net.parameters())
        g = U.flatten(g)
        
        # backtrack line search
        param0 = U.flatten(pi.policy_net.parameters())
        expected_improve = torch.dot(g, pi_lr * g)
        step_size = 1.0
        adv = trajectory['Q'] - trajectory['vpreds']
        success = False # found a good step size
        for _ in range(backtrack_steps):
            newparams = param0 + step_size * pi_lr * g
            U.set_from_flat(pi.policy_net, newparams)

            # new importance sampling gain
            with torch.no_grad():
                newisg = importance_sampling_gain(oldpi, pi, trajectory)

            improve = newisg - isg
            if not torch.isfinite(newisg):
                logger.log('Got infinite importance sampling gain')
            elif newisg.numpy() > (1 + eps) * np.mean(adv):
                logger.log('ISG bigger than the upper bound')
            elif newisg.numpy() < (1 - eps) * np.mean(adv):
                logger.log('ISG smaller than the lower bound')
            elif improve < 0.0:
                logger.log('Policy not improving')
            else:
                logger.log('Step size is OK')
                success = True
                break
            step_size *= 0.5
        else:
            logger.log('Cannot find a good step size')
            U.set_from_flat(pi.policy_net, param0)
        
        logger.record_tabular('Success', success)
        
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
        rolling_buf_episode_rets.extend(trajectory['ep_rets'])
        rolling_buf_episode_lens.extend(trajectory['ep_lens'])
        
        logger.record_tabular("MAEpRet", np.mean(rolling_buf_episode_rets))
        logger.record_tabular('MAEpLen', np.mean(rolling_buf_episode_lens))
        logger.record_tabular('MeanStepRew', np.mean(trajectory['rews']))
        
        logger.dump_tabular()

    pi.save_ckpt(ckpt_dir)
    torch.save(voptimizer, os.path.join(ckpt_dir, 'optim.pth'))
    return

if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    PPO_clip(
        env=env,
        nsteps=64,
        total_timesteps=1024*1000,
        eps = 0.01,
        pi_lr=1e-4,
        v_lr = 1e-4,
        backtrack_steps=10,
        log_dir='/tmp/0/',
        ckpt_dir='/tmp/0/',
        v_iters=3,
        batch_size=64
        )



