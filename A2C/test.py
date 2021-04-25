from cartpole import a2c
from utils import AverageEpisodicRewardTracker

import numpy as np
from numpy.testing import assert_equal


h = {
    "n_iters": 10,
    "eval_interval": 1000,
    "ckpt_interval": 1000,
    "nsteps" : 30, # length of trajectory
    "entropy_coef" : 0.01, #1e-4, 
    "learning_rate": 1e-4,
    "p_coef": 1.0,
    "v_coef":1.0,
    "gamma": 0.99,
    "device": "cpu",
    "nenvs": 16,
    "ckpt_dir": 'ckpt/debug',
    "log_dir": 'log/debug'
}

def test_a2c():
    a2c(h)
    return


def test_reward_tracker():
    batch_reward = np.ones((8, 16)) # nsteps, nenvs
    batch_done = np.zeros((8, 16)).astype(np.bool)
    
    tracker = AverageEpisodicRewardTracker(16)
    
    for _ in range(10):
        tracker.update(batch_reward, batch_done)
    assert_equal(tracker.query(), np.ones(16)*10*8)
    
    # 0'th env of all steps are done
    batch_done[:,0] = True
    tracker = AverageEpisodicRewardTracker(16)
    
    for _ in range(10):
        tracker.update(batch_reward, batch_done)
        
    res = np.ones(16) * 10 * 8
    res[0] = 0
    assert_equal(tracker.query(), res)
    return

if __name__ == '__main__':
    #test_reward_tracker()
    test_a2c()