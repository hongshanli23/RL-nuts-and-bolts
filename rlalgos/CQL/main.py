from dataclasses import dataclass
import torch.nn as nn


@dataclass
class HParams:
    env_name: str
    use_automatic_entropy_tune: bool

class ReplayBuffer:
    pass

class ContinueousPolicy(nn):
    
    

def main(hp: HParams, dataset: ReplayBuffer):
    