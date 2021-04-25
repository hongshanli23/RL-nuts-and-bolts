# components for implementing A2C algorithm

from typing import List, Callable, Tuple, Optional

from collections import namedtuple
import gym
import datetime
import os
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_

Experience = namedtuple('Experience',
        ['curr_state', 
         'action', 
         'log_pi_a',        # has gradient
         'entropy',         # has gradient
         'reward', 
         'v',               # has gradient
         'nx_state', 
         'done'])



class Agent:
    """Agent for A2C 
    Wraps underlying model and its optimizer
    
    """
    def __init__(self, model:nn.Module, h:dict):
        """
        Args:
            h: hyperparamters
        """
        self.h = h
        self.model = model.to(h["device"])
        self.optimizer = optim.Adam(self.model.parameters(), 
            lr=h['learning_rate'])
        self.h = h

    def parse_input(self, x):
        """
        x: input from the env
        """
        if x.ndim == 1: # no batch dim
            x = np.expand_dims(x, axis=0)
        # to torch tensor
        x = x.astype(np.float32)
        x = torch.from_numpy(x).to(self.h['device'])
        return x

    def take_action(self, x:np.ndarray, greedy=False):
        """ Sample an action for the current state x
        
            x: current state
            greedy: use greedy strategy

        return
            (int) integer rep of the action
        """
        x = self.parse_input(x)
        with torch.no_grad():
            proba, _ = self.model(x)

        # sample an action according to
        proba = proba.cpu().numpy()
        
        actions = []
        if greedy: # use in eval
            for i in range(len(proba)):
                actions.append(np.argmax(proba[i]).item())
                
        else:
            for i in range(len(proba)):
                actions.append(np.random.choice(
                    range(self.model.n_actions), size=1, p=proba[i]
                ).item())
                
        return actions


    def get_value(self, x:np.ndarray):
        """ Estimate the state value of the current state x
        
        Args:
            x: current state
            
        return:
            (float) V(s)

        Q: is it better to clip v to certain range ?
        """
        x = self.parse_input(x)
        with torch.no_grad():
            _, v = self.model(x)
        return v.cpu().numpy()
    
    def save_ckpt(self, n_iter):
        """Ckpt the model and optimzer state"""
        ckpt_dir = self.h['ckpt_dir']
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        
        ckpt = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
        }
        fpath = os.path.join(ckpt_dir, f'ckpt-{n_iter}.pt')
        with open(fpath, 'wb') as f:
            torch.save(ckpt, f)
        return
    
    def resume(self, ckpt_file):
        """resume the state of policy /value net and optmizer
        from a checkpoint file
        ckpt_file: path to the ckpt file
        """
        ckpt = torch.load(ckpt_file)
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        return
    
        
    

class TrajectorySampler:
    """Sample a trajectory"""
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.curr_state = self.env.reset()
    
    def __call__(self, nsteps:int):
        trajectory = []
        for _ in range(nsteps):
            x = self.agent.parse_input(self.curr_state)
            action_proba, v = self.agent.model(x)
            
            m = Categorical(action_proba)
            actions = m.sample()
            
            # the env automatically returns the 
            # reset of the env if done == True
            obs, rew, done, _ = self.env.step(actions.numpy())
            
            log_proba = m.log_prob(actions)
            #assert log_proba.requires_grad
            #assert m.entropy().requires_grad
            
            # add to experience
            e = Experience(
                curr_state = self.curr_state,
                action = actions.numpy(),
                reward = rew, 
                log_pi_a = log_proba,        # has grad
                v = v,                       # has grad
                entropy = m.entropy(),       # has grad
                nx_state = obs,
                done = done
            )
            
            trajectory.append(e)
            self.curr_state = obs
        return trajectory

    
def estimate_Q(agent:Agent, trajectory:List[Experience], 
               gamma:float, nenvs:int) -> np.ndarray:
    """Estimate Q(s_t, a_t) at each time step"""
    Q = np.zeros((len(trajectory), nenvs))
    
    R = agent.get_value(trajectory[-1].nx_state) # future reward
    R = np.squeeze(R, axis=1)
    for t in reversed(range(len(trajectory))):
        not_done = ~trajectory[t].done
        Q[t,:]=trajectory[t].reward + gamma * R * not_done
        R = Q[t,:]
    return Q


def one_step_loss(one_step, Qt, device):
    """Compute loss for one step
    Args:
        one_step: one step in trajectory
        Qt: estimate of Q(s_t, a_t) for one step for all parallel env
    
    Return:
        dict
    """    
    Qt = torch.tensor(Qt, dtype=torch.float, device=device)
    Qt = torch.unsqueeze(Qt, dim=1)

    # value loss
    v = one_step.v
    
    assert v.shape == Qt.shape, f"v : {v.shape}, Qt: {Qt.shape}"
    v_loss = F.mse_loss(v, Qt)

    # try torch.detach_(v) here
    # I think it will fail
    # because v_loss is tied to v and its graph
    # if I detach v in place, then v_loss 
    # cannot backpropagate

    # policy loss
    adv = Qt - v.detach()
    log_pi_a = torch.unsqueeze(one_step.log_pi_a, dim=1)
    assert log_pi_a.shape == adv.shape, f"log_pi_a: {log_pi_a.shape}, adv: {adv.shape}"
    p_loss = -torch.mean(log_pi_a * adv)

    return {
        "p_loss": p_loss,
        "v_loss": v_loss,
        "entropy": torch.mean(one_step.entropy), # avg ent across env
        "adv": torch.mean(torch.detach(adv)),
        "log_pi_a": torch.mean(torch.detach(log_pi_a))
    }


def compute_loss(traj, Q, device):
    """Compute loss on the sampled trajectory
    """
    loss_dict = {}
    for t in range(len(traj)):
        one_step_loss_dict = one_step_loss(traj[t], Q[t], device)
        for k, v in one_step_loss_dict.items():
            loss_dict[k] = loss_dict.get(k, 0) + v
    
    # average the loss across time steps
    for k in loss_dict:
        loss_dict[k] = loss_dict[k] / len(traj)
    return loss_dict