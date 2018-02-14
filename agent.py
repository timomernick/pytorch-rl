import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable
import math
import numpy as np
from memory import MemoryBank


class Agent(nn.Module):
    def __init__(self, environment):
        super(Agent, self).__init__()

        self.env = environment

        self.memory_bank = MemoryBank(max_memories=1000)

        self.network = nn.Sequential(
            nn.Linear(environment.observation_space.n, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, environment.action_space.n)
        )

        self.optimizer = optim.SGD(self.parameters(), lr=0.1)

        self.loss_fn = nn.MSELoss()

        def weights_init(m):
            classname = m.__class__.__name__
            print(classname)
            if classname.find('Linear') != -1:
                init.xavier_uniform(m.weight.data)

        self.apply(weights_init)

    def run(self, episode, max_episodes):
        # 'epsilon' is the exploration rate, where 0 means to always use the model to choose actions,
        # and 1 means to always choose a random action.
        # decay from 'epsilon_max' to near 0 as 'episode' approaches 'max_episodes'
        epsilon_max = 0.1
        epsilon = epsilon_max * (1.0 - math.log((episode * (1.0 / max_episodes * 1.7)) + 1.0))

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

        # remember this state transition and optimize the model
        if train:
            self.experience(self.state_to_one_hot(state), action, Q, self.state_to_one_hot(next_state), reward)

        return next_state, reward, done

    def forward(self, state):
        # the model outputs Q, the value of performing each action.
        state = Variable(torch.FloatTensor(state))
        Q = self.network(state)

        # choose the action with the highest value.
        action_val, action_idx = Q.max(dim=1)
        action_idx = action_idx.data[0]
        return action_idx, Q

    def state_to_one_hot(self, state):
        # convert a state integer to a one-hot vector
        # 3 -> [0001000000000000]
        state_one_hot = np.zeros([self.env.observation_space.n], dtype=np.float32)
        state_one_hot[state] = 1.0
        return state_one_hot.reshape(-1, self.env.observation_space.n)

    def get_action(self, state, epsilon):
        # choose an action using the model.
        action, Q = self(self.state_to_one_hot(state))

        # randomly choose an action based on epsilon (exploration rate)
        if np.random.rand(1) < epsilon:
            action = self.env.action_space.sample()

        return action, Q

    def experience(self, state, action, Q, next_state, reward):
        # remember this state transition.
        self.memory_bank.push(state, action, Q, next_state, reward)

        # optimize the model given this transition and past memories.
        self.optimize_model()

    def optimize_model(self):
        # sample a random batch of memories.
        batch_size = 32
        memories = self.memory_bank.sample(batch_size)
        if memories is None:
            return

        # create batches of states, actions, next states, rewards, and Q values.
        states = list()
        actions = list()
        next_states = list()
        rewards = list()
        Qs = list()
        for batch_idx in range(batch_size):
            memory = memories[batch_idx]
            states.append(memory.state)
            actions.append(memory.action)
            next_states.append(memory.next_state)
            rewards.append(memory.reward)
            Qs.append(memory.Q)
        states = np.stack(states, axis=0)
        actions = np.stack(actions, axis=0)
        next_states = np.stack(next_states, axis=0)
        rewards = np.stack(rewards, axis=0)
        Qs = np.stack(Qs, axis=0)

        # get Q values for next states.
        _, Q1 = self(next_states)
        Q1 = Q1.data.cpu().squeeze(1).numpy()

        # get the highest Q values for the next states.
        # this represents the value of being in the next state.
        maxQ1 = np.max(Q1, axis=1)

        # future reward discount factor.
        # this represents the uncertainty of the future reward (maxQ1).
        gamma = 0.99

        # we will train the network to output the same 'Qs' but with the value of the chosen action
        # set to the future discounted reward.
        targetQ = Variable(torch.FloatTensor(Qs))
        for batch_idx in range(batch_size):
            # set the value of the chosen action to the current reward plus the future discounted reward.
            targetQ[batch_idx,actions[batch_idx]] = rewards[batch_idx] + (gamma * maxQ1[batch_idx])

        # get the Q values for the original states.
        _, outputQ = self(states)

        # minimize difference between Q values for original states, and target Q values computed above.
        loss = self.loss_fn(outputQ, targetQ)
        #print("loss", loss.data[0])

        # optimize model.
        self.zero_grad()
        loss.backward()
        self.optimizer.step()

