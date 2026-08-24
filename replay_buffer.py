import numpy as np
import torch


class ReplayBuffer:

    def __init__(self, capacity, state_size, device):

        self.capacity = capacity
        self.state_size = state_size
        self.device = device

        self.position = 0
        self.size = 0

        # State
        # [WK, WP, BK, turn]

        self.states = np.zeros(
            (capacity, state_size),
            dtype=np.float32
        )

        self.next_states = np.zeros(
            (capacity, state_size),
            dtype=np.float32
        )

        # Action
        # 0 - 9

        self.actions = np.zeros(
            capacity,
            dtype=np.int64
        )

        # Reward

        self.rewards = np.zeros(
            capacity,
            dtype=np.float32
        )

        # Done

        self.dones = np.zeros(
            capacity,
            dtype=np.float32
        )

        # Action mask

        # KPK has 10 actions

        self.masks = np.zeros(
            (capacity, 10),
            dtype=np.float32
        )

    # ADD

    def add(self,state,action,reward,next_state,done,mask):

        self.states[self.position] = state

        self.actions[self.position] = action

        self.rewards[self.position] = reward

        self.next_states[self.position] = next_state

        self.dones[self.position] = float(done)

        self.masks[self.position] = mask

        # Move to next position.
        # When capacity is reached, this automatically
        # overwrites the oldest experience.

        self.position = (
            self.position + 1
        ) % self.capacity

        self.size = min(
            self.size + 1,
            self.capacity
        )

    # SAMPLE

    def sample(self, batch_size):

        if self.size < batch_size:
            raise ValueError(
                "Not enough experiences in replay buffer."
            )

        indices = np.random.choice(
            self.size,
            batch_size,
            replace=False
        )

        # Convert to PyTorch tensors

        states = torch.FloatTensor(
            self.states[indices]
        ).to(self.device)

        actions = torch.LongTensor(
            self.actions[indices]
        ).to(self.device)

        rewards = torch.FloatTensor(
            self.rewards[indices]
        ).to(self.device)

        next_states = torch.FloatTensor(
            self.next_states[indices]
        ).to(self.device)

        dones = torch.FloatTensor(
            self.dones[indices]
        ).to(self.device)

        masks = torch.FloatTensor(
            self.masks[indices]
        ).to(self.device)

        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
            masks
        )

    # LENGTH

    def __len__(self):

        return self.size