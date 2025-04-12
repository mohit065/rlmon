import random
from collections import deque, namedtuple
import torch
import numpy as np

Experience = namedtuple('Experience',
                        ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, *args):
        """Saves an experience."""
        self.memory.append(Experience(*args))

    def sample(self, batch_size):
        """Randomly samples a batch of experiences from memory."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def is_ready(self, batch_size):
        return len(self.memory) >= batch_size