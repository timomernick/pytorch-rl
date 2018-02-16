import gym
import math
import numpy as np
from agent_es import ESAgent
from agent_rl import RLAgent

# Load the environment.
env = gym.make("FrozenLake-v0")

# Create an agent to act in the environment.
agent = ESAgent(environment=env, use_cuda=False)

# Train the agent.
max_episodes = 1000
episode_rewards = np.zeros([max_episodes], dtype=np.float32)
for episode in range(max_episodes):
    total_reward = agent.run(episode, max_episodes)

    print("==== Episode " + str(episode) + " ====")

    episode_rewards[episode] = total_reward
    print("Total reward: " + str(total_reward))

    average_reward_max_n = 20
    average_reward_n = min(episode + 1, average_reward_max_n)
    average_reward_values = episode_rewards[(episode+1)-average_reward_n:episode+1]
    average_reward = average_reward_values.mean()
    print("Average reward: " + str(average_reward))

# Test after training.
num_test = 100
num_wins = 0
for test_idx in range(num_test):
    win = agent.test()
    if win:
        num_wins += 1

print("Wins: " + str(num_wins))
print("Losses: " + str(num_test - num_wins))

