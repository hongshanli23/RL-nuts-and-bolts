import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MLPSingleArch(nn.Module):
    def __init__(self, input_dim:int, n_actions:int):
        super(MLPSingleArch, self).__init__()
        self.n_actions = n_actions
        
        def _init(m):
            nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            nn.init.constant_(m.bias, val=0)
            return m
        
        # feature representation
        self.fr = nn.Sequential(
            _init(nn.Linear(input_dim, 256)),
            nn.ReLU(),
            _init(nn.Linear(256, 512)),
            nn.ReLU()
        )

        # policy head
        self.policy = nn.Linear(512, n_actions)
        _init(self.policy)

        # value head
        self.value = nn.Linear(512, 1)
        _init(self.value)

        # init the policy head and value head

    def forward(self, x:torch.tensor):
        # shape of x : (batch_size, input_dim)
        f = self.fr(x)
        return F.softmax(self.policy(f), dim=-1), self.value(f)
    
    