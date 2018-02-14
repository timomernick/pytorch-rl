import numpy as np


class Memory(object):
    def __init__(self, state, action, Q, next_state, reward):
        self.state = state
        self.action = action
        self.Q = Q
        self.next_state = next_state
        self.reward = reward


class MemoryBank(object):
    def __init__(self, max_memories):
        self.max_memories = max_memories
        self.memories = [None] * max_memories
        self.num_memories = 0
        self.memory_index = 0

    def push(self, state, action, Q, next_state, reward):
        memory = Memory(state, action, Q, next_state, reward)
        self.memories[self.memory_index] = memory
        self.num_memories = min(self.num_memories + 1, self.max_memories)
        self.memory_index = (self.memory_index + 1) % self.max_memories

    def sample(self, batch_size):
        if self.num_memories < batch_size:
            return None
        random_indices = np.random.permutation(self.num_memories)[0:batch_size]
        samples = list()
        for idx in random_indices:
            samples.append(self.memories[idx])
        return samples
