# wrappers to gym env

import gym
import numpy as np


class AutoReset(gym.Wrapper):
    """Automatically reset the env when it is done"""

    def __init__(self, env):
        super(AutoReset, self).__init__(env)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done:
            obs = self.env.reset()
        return obs, rew, done, info


class RecoverAction(gym.Wrapper):
    """recover from normalized action
    Actions coming from the policy is normalized to [-1, 1];
    recover it to [action_space.low, action_space.high]
    """

    def __init__(self, env):
        super(RecoverAction, self).__init__(env)

    def step(self, action):
        act_k = (self.action_space.high - self.action_space.low) / 2.
        act_b = (self.action_space.high + self.action_space.low) / 2.

        action = act_k*action + act_b
        obs, rew, done, info = self.env.step(action)
        return obs, rew, done, info


class TransformReward(gym.Wrapper):
    """Apply transformation of rewards"""

    def __init__(self, env, transform_fn):
        super(TransformReward, self).__init__(env)
        self.transform_fn = transform_fn

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        rew = self.transform_fn(rew)
        return obs, rew, done, info


class Truncate(gym.Wrapper):
    """Truncate an infinite episodic task by setting lower bound
    of negative reward
    """

    def __init__(self, env, lower_bound):
        super(Truncate, self).__init__(env)
        self.lower_bound = lower_bound

        self.total_rew = 0.0

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.total_rew += rew
        if self.total_rew < self.lower_bound:
            done = True
            self.total_rew = 0.0
        return obs, rew, done, info


class StartWithRandomActions(gym.Wrapper):
    """ Makes random number of random actions at the beginning of each
    episode. """

    def __init__(self, env, max_random_actions=30):
        super(StartWithRandomActions, self).__init__(env)
        self.max_random_actions = max_random_actions
        self.real_done = True

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.real_done = info.get("real_done", True)
        return obs, rew, done, info

    def reset(self, **kwargs):
        obs = self.env.reset()
        if self.real_done:
            num_random_actions = np.random.randint(
                self.max_random_actions + 1)
            for _ in range(num_random_actions):
                obs, _, _, _ = self.env.step(
                    self.env.action_space.sample())
            self.real_done = False
        return obs
