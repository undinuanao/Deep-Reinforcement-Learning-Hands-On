import gym

e = gym.make('CartPole-v0')

obs = e.reset()
# print(obs)
# print(e.action_space)
# print(e.observation _space)
# print(e.step(0))
# print(e.action_space.sample())
# print(e.observation_space.sample())
print(type(e))