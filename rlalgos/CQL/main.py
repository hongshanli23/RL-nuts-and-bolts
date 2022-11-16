from dataclasses import dataclass
import d4rl
# import gymnasium as gym
import gym
from rlkits.memory import EnvReplayBuffer
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F




@dataclass
class HParams:
    env_name: str
    use_automatic_entropy_tune: bool
    replay_buffer_size: int


def atanh(x: torch.Tensor):
    """compute inverse hyperbolic tangent 
    https://mathworld.wolfram.com/InverseHyperbolicTangent.html
    target is in the range [-1, 1] 
    """
    nu = (1 + x).clamp(1e-6)
    denom = (1 - x).clamp(1e-6)
    return 0.5*torch.log(nu / denom)

class ReplayBuffer:
    pass

## Distributions ##




class ContinueousPolicy(nn.Module):
    def __init__(self,obs_dim:int, action_dim:int,  hidden_sizes:List[int], std:float=None, init_w:float=1e-3, device="cpu"):

        super().__init__()
        self.log_std = None
        self.std = std

        self.device = device
        
        # backbone nn
        self.fcs = []
        in_size = obs_dim
        for hs in hidden_sizes:
            self.fcs.append(
                nn.Linear(in_size, hs)
            )
            in_size = hs
            # TODO try layer norm
        
        # TODO try some initialization tricks 
        self.last_fc = nn.Linear(in_size, action_dim)
        
        if std is None:
            # then learn it
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)

        self = self.to(self.device)

    def forward(self, inputs: torch.Tensor):
        h = inputs
        for layer in self.fcs:
            h = layer(h)
            h = F.relu(h) 
        return self.last_fc(h)

    def get_actions(self, obs: np.ndarray, deterministic: bool)->torch.Tensor:
        """
        obs: batch of observations packed as numpy array
        deterministic: whether actions should be deterministic 
        """
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        return self(obs)

    def log_prob(self, obs, actions):
        """compute log probability of actions
        We model continuous action as a Gaussian distribution with mean and a std
        """
        

    
    

def main(hp: HParams):
    env = gym.make(hp.env_name)
    buff = EnvReplayBuffer(hp.replay_buffer_size, env=env)

    d = env.get_dataset()
    print(d)

    return

if __name__ == "__main__":
    hp = HParams(
        env_name="hopper-medium-v2", 
        use_automatic_entropy_tune=True,
        replay_buffer_size=int(2e3)) 
    
    # generate 

    main(hp)