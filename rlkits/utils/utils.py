import numpy as np
from typing import Iterable, List, Tuple
import torch

import matplotlib.pyplot as plt
from IPython import display


def flatten(params: Tuple[torch.Tensor, ...])-> torch.Tensor:
    """Convert input tensors into a flat vector"""
    return torch.cat([p.view(-1) for p in params])
def set_from_flat(net, flat_vector):
    """Set parameters of a nn from a flat vector"""
    prev_ix = 0
    for p in net.parameters():
        sz = int(np.prod(list(p.size())))
        p.data.copy_(flat_vector[prev_ix:prev_ix+sz].view(p.size()))
        prev_ix += sz
    return
def explained_variance(ypred,y):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary

def inspect_agent(env, agent, nsteps=1000):
    """Look at how the agent behaves in the 
    env for nsteps time steps
    """
    # see actions 
    ob = env.reset()
    total_rew = 0.0
    for t in range(nsteps):
        show_state(env, t)
        ac, *_ = agent.step(ob)
        ob, rew, new, _ = env.step(ac)
        total_rew += rew
    
    # clear the last image
    plt.clf()
    print('total reward: ', total_rew)
    print('avg reward: ', total_rew / nsteps)
    return

def show_state(env, step=0, info=""):
    plt.figure(3)
    plt.clf() # clear current figure
    plt.imshow(env.render(mode='rgb_array'))
    plt.axis('off')
    display.clear_output(wait=True)
    display.display(plt.gcf()) # get_current figure

