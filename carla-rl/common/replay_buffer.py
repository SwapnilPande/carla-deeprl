""" 
Adapted from PyTorch Lightning examples:
https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/master/pl_bolts/models/rl/common/memory.py
"""

from collections import deque, namedtuple

import numpy as np
import torch

Experience = namedtuple('Experience', field_names=['obs', 'action', 'reward', 'next_obs', 'done'])

class ReplayBuffer:
    def __init__(self, capacity=int(1e6)):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        obs, actions, rewards, next_obs, dones = zip(*[self.buffer[idx] for idx in indices])
        return (
            torch.FloatTensor(obs).reshape(batch_size, -1).cuda(),
            torch.FloatTensor(actions).reshape(batch_size, -1).cuda(),
            torch.FloatTensor(rewards).reshape(batch_size, -1).cuda(),
            torch.FloatTensor(next_obs).reshape(batch_size, -1).cuda(),
            torch.FloatTensor(dones).reshape(batch_size, -1).cuda()
        )


class PERBuffer(ReplayBuffer):
    def __init__(self, buffer_size, prob_alpha=0.6, beta_start=0.4, beta_frames=100000, priority_append=True):
        super().__init__(capacity=buffer_size)
        self.beta_start = beta_start
        self.beta = beta_start
        self.beta_frames = beta_frames
        self.prob_alpha = prob_alpha
        self.capacity = buffer_size
        self.pos = 0
        self.buffer = []
        self.priorities = np.zeros((buffer_size, ), dtype=np.float32)
        self.priority_append = priority_append

    def update_beta(self, step):
        """
        Update the beta value which accounts for the bias in the PER

        Args:
            step: current global step

        Returns:
            beta value for this indexed experience
        """
        beta_val = self.beta_start + step * (1.0 - self.beta_start) / self.beta_frames
        self.beta = min(1.0, beta_val)

        return self.beta

    def append(self, exp) -> None:
        """
        Adds experiences from exp_source to the PER buffer

        Args:
            exp: experience tuple being added to the buffer
        """
        # what is the max priority for new sample
        max_prio = self.priorities.max() if self.buffer else 1.0

        if self.priority_append:
            if len(self.buffer) < self.capacity:
                self.buffer.append(exp)
                self.priorities[self.pos] = max_prio
                self.pos = (self.pos + 1) % self.capacity
            else:
                self.pos = np.argmin(self.priorities)
                self.buffer[self.pos] = exp
                self.priorities[self.pos] = max_prio
        else:
            if len(self.buffer) < self.capacity:
                self.buffer.append(exp)
            else:
                self.buffer[self.pos] = exp

            # the priority for the latest sample is set to max priority so it will be resampled soon
            self.priorities[self.pos] = max_prio

            # update position, loop back if it reaches the end
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size=32):
        """
        Takes a prioritized sample from the buffer

        Args:
            batch_size: size of sample

        Returns:
            sample of experiences chosen with ranked probability
        """
        # get list of priority rankings
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        # probability to the power of alpha to weight how important that probability it, 0 = normal distirbution
        probs = prios**self.prob_alpha
        probs /= probs.sum()

        # choise sample of indices based on the priority prob distribution
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        # samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])

        samples = (
            torch.FloatTensor(states).reshape(batch_size, -1).cuda(),
            torch.FloatTensor(actions).reshape(batch_size, -1).cuda(),
            torch.FloatTensor(rewards).reshape(batch_size, -1).cuda(),
            torch.FloatTensor(next_states).reshape(batch_size, -1).cuda(),
            torch.FloatTensor(dones).reshape(batch_size, -1).cuda()
        )
        total = len(self.buffer)

        # weight of each sample datum to compensate for the bias added in with prioritising samples
        weights = (total * probs[indices])**(-self.beta)
        weights /= weights.max()

        # return the samples, the indices chosen and the weight of each datum in the sample
        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, batch_indices, batch_priorities) -> None:
        """
        Update the priorities from the last batch, this should be called after the loss for this batch has been
        calculated.

        Args:
            batch_indices: index of each datum in the batch
            batch_priorities: priority of each datum in the batch
        """
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio
