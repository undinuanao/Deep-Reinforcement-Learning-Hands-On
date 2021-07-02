#!/usr/bin/env python3
import gym
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim


HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)


Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


def iterate_batches(env, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    sm = nn.Softmax(dim=1)
    while True:
        obs_v = torch.FloatTensor([obs])
        # print("obs_v:", obs_v)
        # act_prob_test = net(obs_v)
        # print("未经过Softmax处理过的网络输出：",act_prob_test)
        act_probs_v = sm(net(obs_v))
        # print("Softmax处理后的网络输出:", act_probs_v)
        act_probs = act_probs_v.data.numpy()[0] #将tensor类型的张量转换为numpy里的ndarray类型
        # print("动作的概率act_probs是：{0}, 其数据类型是：{1}".format(act_probs, type(act_probs)))
        action = np.random.choice(len(act_probs), p=act_probs)
        # print("选择的动作ACTION：", action)
        next_obs, reward, is_done, _ = env.step(action)
        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs, action=action))
        if is_done:
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs


def filter_batch(batch, percentile):
    rewards = list(map(lambda s: s.reward, batch))
    # print("所有的奖励为：{0},共有{1}个".format(rewards,len(rewards)))
    reward_bound = np.percentile(rewards, percentile)
    # print("奖励边界为：", reward_bound)
    reward_mean = float(np.mean(rewards))
    # print("奖励均值为：", reward_mean)

    train_obs = []
    train_act = []
    for example in batch:
        if example.reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, example.steps))
        # print("样本中的观察train_obs为:",train_obs[0::20])
        train_act.extend(map(lambda step: step.action, example.steps))
        # print("样本中的动作train_act为:",train_act[0::20])
    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean


if __name__ == "__main__":
    for i in range(10):
        env = gym.make("CartPole-v0")
        # env = gym.wrappers.Monitor(env, directory="mon", force=True)
        obs_size = env.observation_space.shape[0]
        n_actions = env.action_space.n

        net = Net(obs_size, HIDDEN_SIZE, n_actions)
        objective = nn.CrossEntropyLoss()
        optimizer = optim.Adam(params=net.parameters(), lr=0.01)
        writer = SummaryWriter(comment="-cartpole")

    
        for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
            obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
            optimizer.zero_grad()
            action_scores_v = net(obs_v)
            loss_v = objective(action_scores_v, acts_v)
            loss_v.backward()
            # env.render()
            optimizer.step()
            print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (
                iter_no, loss_v.item(), reward_m, reward_b))
            print(loss_v)
            writer.add_scalar("loss", loss_v.item(), iter_no)
            writer.add_scalar("reward_bound", reward_b, iter_no)
            writer.add_scalar("reward_mean", reward_m, iter_no)
            if reward_m > 199:
                print("Solved!")
                break
        writer.close()
