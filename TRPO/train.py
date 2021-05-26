from main import TRPO
import os
import json
import torch
import gym

from rlkits.env_batch import ParallelEnvBatch


def main():
    # h = "/opt/ml/input/config/hyperparameters.json"
    # with open(h, 'r') as f:
    # h = json.load(f)
    
    def make_env():
        return gym.make('Pendulum-v0')
    
    env = ParallelEnvBatch(make_env, nenvs=2)
    
    TRPO(
        env=env,
        nsteps=128,
        total_timesteps=2*128*1000, 
        log_dir='/home/ubuntu/logs/trpo/pendulum/256_512'
        ckpt_dir='/home/ubuntu/logs/trpo/pendulum/256_512',
        hidden_layers=[256, 512],
        activation=torch.nn.ReLU,
        )
    env.close()
    return
    




if __name__ == '__main__':
    main()
