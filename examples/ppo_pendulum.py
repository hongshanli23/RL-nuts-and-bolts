
import argparse
from rlkits.env_batch import ParallelEnvBatch
from rlkits.env_wrappers import AutoReset, StartWithRandomActions
from rlalgos.PPO.ppo import PPO
import gym
import torch


parser = argparse.ArgumentParser()
parser.add_argument('--nenvs', type=int, default=4,
                   help='number of parallel envs')
parser.add_argument('--niters', type=int, default=10000,
                   help='number of training iterations')
parser.add_argument('--nsteps', type=int, default=128,
                   help='length of sampled trajectory')
# Algo Hyp
parser.add_argument('--max-kl', type=float, default=1e-3,
                   help='KL target')
parser.add_argument('--eps', type=float, default=0.1,
                   help='clip range')
parser.add_argument('--beta', type=float, default=0.5,
                   help='Penalty factor for KL')
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
parser.add_argument('--batch-size', type=int, default=None,
                   help='batch size of each policy update')
parser.add_argument('--epochs', type=int, default=3,
                   help='Number of epochs to train on one trajectory')
parser.add_argument('--log-intervals', type=int, default=10,
                   help='logging frequency')
parser.add_argument('--log-dir', type=str, default="/tmp/ppo",
                   help='directory to write logs')
parser.add_argument('--ckpt-dir', type=str, default='/tmp/ppo',
                   help='directory to write checkpoint')
args = parser.parse_args()


def make_env():
    env = gym.make('Pendulum-v0')
    env = AutoReset(env)
    env = StartWithRandomActions(env, max_random_actions=5)
    return env

env=ParallelEnvBatch(make_env, nenvs=args.nenvs)

PPO(
    env=env,
    nsteps=args.nsteps,
    total_timesteps=args.nenvs*args.nsteps*args.niters,
    max_kl=args.max_kl,
    beta=args.beta,
    eps = args.eps,
    gamma= args.gamma,
    pi_lr= args.pi_lr,
    v_lr = args.v_lr,
    ent_coef=args.ent_coef,
    epochs=args.epochs,
    batch_size=args.batch_size or args.nsteps * args.nenvs,
    log_interval=args.log_intervals,
    max_grad_norm=args.max_grad_norm,
    reward_transform=None,
    log_dir=args.log_dir,
    ckpt_dir=args.ckpt_dir,
    hidden_layers=[256, 256, 64],
    activation=torch.nn.ReLU,
    )

env.close()
