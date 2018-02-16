import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from agent import Agent


class ESAgent(Agent):
    def __init__(self, environment, use_cuda):
        super(ESAgent, self).__init__(environment=environment, use_cuda=use_cuda)

        self.epsilon_max = 0.0

        self.env = environment

        self.network = nn.Sequential(
            nn.Linear(environment.observation_space.n, 100),
            nn.Tanh(),
            nn.Linear(100, environment.action_space.n)
        )

        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Linear') != -1:
                m.weight.data.uniform_(-1, 1)

        self.apply(weights_init)

        if self.use_cuda:
            self.network = self.network.cuda()

    def run(self, episode, max_episodes):
        # how many critters
        num_population = 100

        # std dev of parameter noise
        sigma = 0.1

        # learning rate
        alpha = 0.01

        # run this many episodes per critter
        num_inner_episodes = 1

        # start with the current parameters as the best known
        bestW = nn.utils.parameters_to_vector(self.network.parameters()).data.cpu().numpy()

        # reward per population
        R = np.zeros([num_population])

        # parameter noise per population.
        N = np.random.randn(num_population, bestW.shape[0])

        for pop_idx in range(num_population):
            # add noise to parameters.
            w_try_cpu = bestW + (sigma * N[pop_idx])
            w_try = self.optional_cuda(Variable(torch.FloatTensor(w_try_cpu)))
            nn.utils.vector_to_parameters(w_try, self.network.parameters())

            # run with these weights and remember reward
            for inner_episode in range(num_inner_episodes):
                reward = super(ESAgent, self).run(episode, max_episodes)
                R[pop_idx] += reward

        # update weights if there is any reward signal.
        R_std = R.std()
        if R_std > 1e-8:
            # move parameters in the direction of N proportional to reward
            A = (R - R.mean()) / R_std
            bestW += (alpha / (num_population * sigma)) * np.dot(N.T, A)

            # update the model parameters with the new bestW
            bestW = self.optional_cuda(Variable(torch.FloatTensor(bestW)))
            nn.utils.vector_to_parameters(bestW, self.network.parameters())

        # now that the model weights are updated, run an episode and return the reward.
        return super(ESAgent, self).run(episode, max_episodes)

    def experience(self, state, action, Q, next_state, reward):
        # ES learns after the episode is finished based on the total reward.
        # so we don't need to remember state transitions or learn every step.
        return
