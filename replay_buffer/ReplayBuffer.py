"""
A vanilla replay buffer used to store MDP tuples for a RL agent.
# TODO cite git of inspiration
"""

# native packages
from collections import namedtuple, deque
import random

# third party modules
import numpy as np
import torch

# own modules


class ReplayBuffer:

    def __init__(self, buffer_size, batch_size, seed, device=None):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = torch.device(device)

    def add_experience(self, states, actions, rewards, next_states, dones):
        experience = self.experience(states, actions, rewards, next_states, dones)
        self.memory.append(experience)

    def sample(self,batch_size):
        experiences = self.pick_experiences(batch_size)
        states, actions, rewards, next_states, dones = self.separate_out_data_types(experiences)
        return states, actions, rewards, next_states, dones

    def separate_out_data_types(self, experiences):
        """Puts the sampled experience into the correct format for a PyTorch neural network"""
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            self.device)
        dones = torch.from_numpy(np.vstack([int(e.done) for e in experiences if e is not None])).float().to(self.device)

        return states, actions, rewards, next_states, dones

    def pick_experiences(self,batch_size):
        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)