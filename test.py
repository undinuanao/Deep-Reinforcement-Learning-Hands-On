#!/usr/bin/env python3
import gym, gym.spaces
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim


HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70


class DiscreteOneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(DiscreteOneHotWrapper, self).__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete)
        self.observation_space = gym.spaces.Box(0.0, 1.0, (env.observation_space.n, ), dtype=np.float32)

    def observation(self, observation):
        res = np.copy(self.observation_space.low)
        res[observation] = 1.0
        return res

if __name__ == "__main__":
    e1 = gym.make("FrozenLake-v0")
    env = DiscreteOneHotWrapper(gym.make("FrozenLake-v0"))
    print("e1.actionspace_space 等于 {0}, 实例为：{1}".format(e1.action_space, e1.action_space.sample()))
    print("env.actionspace_space 等于 {0}, 实例为：{1}".format(env.action_space, env.action_space.sample()))
    print("e1.actionspace_space 等于 {0}, 实例为：{1}".format(e1.action_space, e1.action_space.sample()))
    print("e1.actionspace_space 等于 {0}, 实例为：{1}".format(e1.action_space, e1.action_space.sample()))
