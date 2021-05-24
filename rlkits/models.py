import torch
import torch.nn as nn
import math

def ortho_init(m: nn.Module, init_scale=math.sqrt(0.5)):
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
                ortho_init(nn.Linear(prev, h, bias=False))
            )
            layers.append(activation())
            prev = h

        layers.append(
            ortho_init(nn.Linear(prev, output_shape, bias=False)
            ))
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)
