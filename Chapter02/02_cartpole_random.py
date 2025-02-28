import gym


if __name__ == "__main__":
    env = gym.make("CartPole-v0")

    total_reward = 0.0
    total_steps = 0
    obs = env.reset()

    while True:
        action = env.action_space.sample()
        env.render()
        obs, reward, done, _ = env.step(action)
        print(obs, reward, done)
        total_reward += reward
        total_steps += 1
        # if total_steps >= 100:
        if done:
            break

    print("Episode done in %d steps, total reward %.2f" % (total_steps, total_reward))
