import argparse
from rlkits.env_batch import ParallelEnvBatch
from rlkits.env_wrappers import AutoReset, StartWithRandomActions
from rlalgos.DDPG.ddpg import DDPG
import gym
import torch


parser = argparse.ArgumentParser()
parser.add_argument('--nenvs', type=int, default=4,
                   help='number of parallel envs')
parser.add_argument('--niters', type=int, default=10000,
                   help='number of training iterations')
parser.add_argument('--nsteps', type=int, default=128,
                   help='length of sampled trajectory')
parser.add_argument('--nupdates', type=int, default=100,
                   help='number of updates to perform between sampling new experiences')
parser.add_argument('--buf-size', type=int, default=10000,
                   help='replay buffer size')

parser.add_argument('--model-update-frequency', type=int, default=5,
                   help='number of iterations between updating target models')

parser.add_argument('--polyak', type=float, default=0.5, 
                   help='linear intepolation factor used to update current model')
parser.add_argument('--gamma', type=float, default=0.99, 
                   help='discount factor')
parser.add_argument('--pi-lr', type=float, default=1e-4,
                   help='policy learning rate')
parser.add_argument('--v-lr', type=float, default=1e-4, 
                   help='value net learning rate')
parser.add_argument('--max-grad-norm', type=float, default=0.1,
                   help='maximum norm of gradients in each nn layer')
parser.add_argument('--batch-size', type=int, default=128,
                   help='batch size of each policy update')
parser.add_argument('--log-interval', type=int, default=1,
                   help='logging frequency')
parser.add_argument('--log-dir', type=str, default="/tmp/ddpg",
                   help='directory to write logs')
parser.add_argument('--ckpt-dir', type=str, default='/tmp/ddpg',
                   help='directory to write checkpoint')
parser.add_argument('--hidden-layers', nargs="+", type=int, 
                    default=[256, 256, 64],
                   help='hidden layers')

args = parser.parse_args()

DDPG(
    env_name='Pendulum-v0',
    nsteps=args.nsteps,
    nenvs=args.nenvs,
    niters=args.niters,
    nupdates=args.nupdates,
    buf_size=args.buf_size, 
    gamma=args.gamma,
    pi_lr=args.pi_lr,
    v_lr=args.v_lr,
    polyak=args.polyak,
    model_update_frequency=args.model_update_frequency,
    batch_size=args.batch_size,
    log_interval=args.log_interval,
    max_grad_norm=args.max_grad_norm,
    log_dir=args.log_dir,
    ckpt_dir=args.ckpt_dir,
    hidden_layers=args.hidden_layers,
)


