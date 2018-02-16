import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from agent import Agent
from memory import MemoryBank

class RLAgent(Agent):
    def __init__(self, environment, use_cuda=False):
        super(RLAgent, self).__init__(environment=environment, use_cuda=use_cuda)

        self.memory_bank = MemoryBank(max_memories=1000)

        self.network = nn.Sequential(
            nn.Linear(environment.observation_space.n, 100),
            nn.Tanh(),
            nn.Linear(100, environment.action_space.n)
        )

        self.optimizer = optim.SGD(self.parameters(), lr=0.1)

        self.loss_fn = nn.MSELoss()

        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Linear') != -1:
                nn.init.xavier_uniform(m.weight.data)

        self.apply(weights_init)

    def experience(self, state, action, Q, next_state, reward):
        # remember this state transition.
        self.memory_bank.push(state, action, Q, next_state, reward)

        # optimize the model given this transition and past memories.
        self.optimize_model()

    def optimize_model(self):
        # sample a random batch of memories.
        batch_size = 100
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
        targetQ = self.optional_cuda(Variable(torch.FloatTensor(Qs)))
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

