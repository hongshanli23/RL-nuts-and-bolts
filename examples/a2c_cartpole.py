import argparse
from rlkits.env_batch import ParallelEnvBatch
from rlkits.env_wrappers import AutoReset, StartWithRandomActions
from rlalgos.A2C.a2c import A2C
import gym
import torch


parser = argparse.ArgumentParser()
parser.add_argument('--nenvs', type=int, default=8,
                   help='number of parallel envs')
parser.add_argument('--nsteps', type=int, default=128,
                   help='length of sampled trajectory')
parser.add_argument('--total-timesteps', type=int, default=int(1e6),
                    help='total number of experiences to sample')
# Algo Hyperparams
parser.add_argument('--gamma', type=float, default=0.99, 
                   help='discount factor')
parser.add_argument('--pi-lr', type=float, default=1e-4,
                   help='policy learning rate')
parser.add_argument('--v-lr', type=float, default=1e-4, 
                   help='value net learning rate')
parser.add_argument('--ent-coef', type=float, default=1e-2,
                   help='entropy coefficient')
parser.add_argument('--max-grad-norm', type=float, default=0.1,
                   help='maximum norm of gradients in each nn layer')
parser.add_argument('--log-intervals', type=int, default=10,
                   help='logging frequency')
parser.add_argument('--ckpt-dir', type=str, default='/tmp/a2c',
                   help='directory to write checkpoint')
parser.add_argument('--clip-episode', action='store_true', 
                    help='clip episode length to 200 steps')
args = parser.parse_args()


A2C(
    env_name='CartPole-v0',
    nsteps=args.nsteps,
    nenvs=args.nenvs,
    total_timesteps=args.total_timesteps,
    gamma= args.gamma,
    pi_lr= args.pi_lr,
    v_lr = args.v_lr,
    ent_coef=args.ent_coef,
    log_interval=args.log_intervals,
    max_grad_norm=args.max_grad_norm,
    reward_transform=None,
    clip_episode=args.clip_episode,
    ckpt_dir=args.ckpt_dir,
    hidden_layers=[256, 256, 64],
    activation=torch.nn.ReLU, 
    )
