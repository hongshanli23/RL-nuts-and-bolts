from collections import defaultdict
from typing import List
import numpy as np
import os
import pickle
from torch.utils.tensorboard import SummaryWriter


class AverageEpisodicRewardTracker:
    """Keep track of average episodic reward for each individual env"""
    def __init__(self, nenvs):
        self.nenvs = nenvs 
        # moving average of all envs
        self.moving_average = np.zeros(nenvs, dtype=np.float32)
        
        # total reward of the current episode of all envs
        self.total_reward = np.zeros(nenvs, dtype=np.float32)
        
        # finished episodes
        self.nep = np.zeros(nenvs, dtype=np.float32) #[5, 4, ] -> [5, 5]
        
    def update(self, batch_reward:np.ndarray, batch_done:np.ndarray):
        """
        batch_reward: 
            shape : (nsteps, nenv), dtype : np.float32
            reward for all envs at each time step in a trajectory
            
        batch_done: 
            shape: (nsteps, nenv), dtype : bool 
            finished or not of all envs at each time step in a trajectory
        """
        assert batch_done.dtype == np.bool
        
        for r, d in zip(batch_reward, batch_done):    
            # only update the moving avarage of the env
            # that are done
            nx_nep = self.nep[d] + 1
            
            self.moving_average[d] = (
                self.moving_average[d] * self.nep[d] + \
                self.total_reward[d]) / nx_nep
            
            self.nep[d] = nx_nep
            # update total reward of the current episode
            self.total_reward += (r * ~d)
            
            # set total reward of the finished episode to 0
            self.total_reward[d] = 0
        return 
    
    def query(self):
        """query moving average of reward of all envs
        if an env never finishes up to this point, 
        return its total reward
        """
        return self.moving_average + self.total_reward
    


class Logger:
    """Log metrics during an experiment"""
    def __init__(self, log_dir:str):
        self.log_dir = log_dir # unique for each experiment
        self._data = defaultdict(list)
        self.file_path = os.path.join(self.log_dir,
                    'experiment-data.pkl')
        
        self.tb = SummaryWriter(log_dir=self.log_dir)

    def add_scalar(self, key, value, n_iter):
        self._data[key].append((n_iter, value))
        self.tb.add_scalar(key, value, n_iter)
        return
    
    def write_object(self, key, value):
        self._data[key].append(value)
        return
    
    def save(self):
        with open(self.file_path, 'wb') as f:
            pickle.dump(self._data, f)
        return

    def load(self):
        """Load metrics after the experiment is done"""
        with open(self.file_path, 'rb') as f:
            data = pickle.load(f)
        return data

def fetch_from_s3():
    pass

    
    