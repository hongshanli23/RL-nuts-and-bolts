import torch
import torch.nn as nn
import math

def ortho_init(m: nn.Module, init_scale=math.sqrt(2)):
    """In place orthogonal initialization"""
    nn.init.orthogonal_(m.weight, gain=init_scale)
    #torch.nn.init.sparse_(m.weight, sparsity=0.9, std=0.01)
    if m.bias is not None:
        nn.init.constant_(m.bias, 0)
    return m


class MLP(nn.Module):
    def __init__(self, *, input_shape, output_shape, hidden_layers,
                 activation=torch.nn.Tanh):
        super(MLP, self).__init__()
        if isinstance(input_shape, tuple):
            input_shape = input_shape[0]
        
        layers = []
        prev = input_shape
        for h in hidden_layers:
            layers.append(
                ortho_init(nn.Linear(prev, h))
            )
            layers.append(activation())
            prev = h

        layers.append(
            ortho_init(nn.Linear(prev, output_shape)
            ))
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)

    
class MLP2heads(nn.Module):
    """A neural network with 2 heads
    Used for actor-critic agent with a single backbone
    """
    def __init__(self, *, input_shape, policy_output_shape, 
                 value_output_shape, 
                 hidden_layers,
                activation=torch.nn.Tanh):
        super(MLP2heads, self).__init__()
        
        layers = []
        
        prev = input_shape
        for h in hidden_layers:
            layers.append(
                ortho_init(nn.Linear(prev, h))
            )
            layers.append(activation())
            prev = h

        self.layers = nn.Sequential(*layers)
        self.policy_head = ortho_init(
            nn.Linear(prev, policy_output_shape)
        )
        self.value_head = ortho_init(
            nn.Linear(prev, value_output_shape)
        )
        
    def forward(self, x):
        y = self.layers(x)
        return self.policy_head(y), self.value_head(y)