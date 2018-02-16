import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable
import math
import numpy as np
from memory import MemoryBank


class Agent(nn.Module):
    def __init__(self, environment, use_cuda=False):
        super(Agent, self).__init__()

        self.env = environment

        self.use_cuda = use_cuda

        self.epsilon_max = 0.1

        self.use_cuda = use_cuda

    def optional_cuda(self, variable):
        if self.use_cuda:
            return variable.cuda()
        else:
            return variable

    def run(self, episode, max_episodes):
        # 'epsilon' is the exploration rate, where 0 means to always use the model to choose actions,
        # and 1 means to always choose a random action.
        # decay from 'epsilon_max' to near 0 as 'episode' approaches 'max_episodes'
        epsilon = self.epsilon_max * (1.0 - math.log((episode * (1.0 / max_episodes * 1.7)) + 1.0))

        total_reward = 0.0

        # reset environment and observe initial state.
        state = self.env.reset()

        # step until the time limit runs out or the agent is done.
        for step in range(self.env.spec.timestep_limit):
            state, reward, done = self.step(state, epsilon=epsilon, train=True)
            total_reward += reward
            if done:
                break

        return total_reward

    def test(self):
        # reset environment and observe initial state.
        state = self.env.reset()

        # step until the time limit runs out or the agent is done.
        for step in range(self.env.spec.timestep_limit):
            # render the environment.
            self.env.render()

            # step and get next state
            state, reward, done = self.step(state, epsilon=0, train=False)

            # win/lose?
            if done:
                if reward > 0:
                    win = True
                else:
                    win = False
                break

        return win

    def step(self, state, epsilon, train):
        # choose an action given the state
        action, Q = self.get_action(state, epsilon)
        Q = Q.data.cpu().squeeze(0).numpy()

        # perform the action and observe the next state
        next_state, reward, done, _ = self.env.step(action)

        # reward/penalty for win/lose
        if done:
            if reward == 0:
                reward = -1 # lose
            else:
                reward = 1 # win
        else:
            reward = -0.001

        # remember this state transition and optimize the model
        if train:
            self.experience(self.state_to_one_hot(state), action, Q, self.state_to_one_hot(next_state), reward)

        return next_state, reward, done

    def forward(self, state):
        # the model outputs Q, the value of performing each action.
        state = self.optional_cuda(Variable(torch.FloatTensor(state)))
        Q = self.network(state)

        # choose the action with the highest value.
        action_val, action_idx = Q.max(dim=1)
        action_idx = action_idx.data[0]
        return action_idx, Q

    def get_action(self, state, epsilon):
        # choose an action using the model.
        action, Q = self(self.state_to_one_hot(state))

        # randomly choose an action based on epsilon (exploration rate)
        if np.random.rand(1) < epsilon:
            action = self.env.action_space.sample()

        return action, Q

    def state_to_one_hot(self, state):
        # convert a state integer to a one-hot vector
        # 3 -> [0001000000000000]
        state_one_hot = np.zeros([self.env.observation_space.n], dtype=np.float32)
        state_one_hot[state] = 1.0
        return state_one_hot.reshape(-1, self.env.observation_space.n)

    def experience(self, state, action, Q, next_state, reward):
        # subclass must implement this to learn from the state transition
        raise NotImplementedError
