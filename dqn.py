import torch
import torch.nn as nn

class DQN(nn.Module):

    def __init__(self,state_size=4,action_size=10):

        super(DQN, self).__init__()

        self.network = nn.Sequential(

            nn.Linear(state_size, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, action_size)
        )

    def forward(self, state):

        return self.network(state)