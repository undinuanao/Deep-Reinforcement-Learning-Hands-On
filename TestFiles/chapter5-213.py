import collections
from os import write
import gym
from tensorboardX import SummaryWriter, writer

ENV_NAME = "FrozenLake-v0"
GAMMA = 0.9
TEST_EPISODES = 20


class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()       
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.values = collections.defaultdict(float)

    def play_n_random_steps(self, count):
        for _ in range(count):
            action = self.env.action_space.sample()
            new_state, reward, is_done, _ = self.env.step(action)
            self.rewards[(self.state, action, new_state)] = reward
            self.transits[(self.state, action)][new_state] += 1
            self.state = self.env.reset() if is_done else new_state

    def cal_action_value(self, state, action):
        target_counts = self.transits[(state, action)]
        total = sum(target_counts.values())
        action_value = 0.0
        for target_state, count in target_counts.items():
            reward = self.rewards[(state, action, target_state)]
            action_value += (count/total) * (reward + GAMMA * self.values[target_state])
        return action_value


    def select_action(self, state):
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.cal_action_value(state, action)
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        while True:
            action = self.select_action(state)
            new_state, reward, is_done, _ = env.step(action)
            self.rewards[(state, action, new_state)] = reward
            self.transits[(state, action)][new_state] += 1
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward

    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            state_values = [self.cal_action_value(state, action)
                            for action in range(self.env.action_space.n)]
            self.values[state] = max(state_values)

    def show_tables(self, table_length):
        reward_ite = iter(self.rewards.items())
        trans_ite = iter(self.transits.items())
        value_ite = iter(self.values.items())
        display_length = table_length
        for i in range(display_length):
            _, __ = next(reward_ite)
            print("Reward:key = {0}, value = {1}".format(_, __))
        for i in range(display_length):
            _, __ = next(trans_ite)
            print("Transition:key = {0}, value = {1}".format(_, __))
        for i in range(display_length):
            try:
                _, __ = next(value_ite)
                print("Value:key = {0}, value = {1}".format(_, __))
            except:
                print(self.values)
                break

    
if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    # writer = SummaryWriter(comment="-v-iteration")
    agent.play_n_random_steps(100)
    agent.show_tables(3)
    agent.value_iteration()
    agent.show_tables(3)


    # agent.value_iteration()


    # writer.close()
