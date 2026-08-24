import copy
import numpy as np
import torch
import torch.optim as optim

from dqn import DQN
from replay_buffer import ReplayBuffer


class DQNAgent:

    def __init__(self,state_size,action_size,device):

        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        # DQN PARAMETERS

        self.gamma = 0.99

        self.learning_rate = 1e-3

        # Epsilon-greedy parameters

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995

        # Train after this many experiences

        self.batch_size = 64

        # Update target network after this many training steps

        self.target_update_freq = 1000

        # NETWORKS

        # Online network

        self.q_network = DQN(
            state_size,
            action_size
        ).to(self.device)

        # Target network

        self.target_network = copy.deepcopy(
            self.q_network
        ).to(self.device)

        self.target_network.eval()

        # OPTIMIZER
    
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=self.learning_rate
        )

        # REPLAY BUFFER

        self.buffer = ReplayBuffer(
            capacity=100000,
            state_size=state_size,
            device=device
        )

        # TRAINING INFORMATION
     
        self.steps = 0
        self.episodes = 0

        self.losses = []

    # SELECT ACTION

    def select_action(self,state,valid_actions,training=True):

        # Exploration

        if training and np.random.random() < self.epsilon:

            valid_indices = np.where(
                valid_actions > 0
            )[0]

            if len(valid_indices) == 0:
                return 0

            return int(
                np.random.choice(
                    valid_indices
                )
            )
        
        # Exploitation

        with torch.no_grad():

            state_tensor = torch.FloatTensor(
                state
            ).unsqueeze(0).to(self.device)

            q_values = self.q_network(
                state_tensor
            )

            q_values = q_values.cpu().numpy()[0]

        # Remove illegal actions

        q_values = np.where(
            valid_actions > 0,
            q_values,
            -np.inf
        )

        return int(
            np.argmax(q_values)
        )
    
    # STORE TRANSITION

    def store_transition(self,state,action,reward,next_state,done,mask):

        self.buffer.add(
            state,
            action,
            reward,
            next_state,
            done,
            mask
        )

    # TRAIN

    def train_step(self):

        # Not enough experiences

        if len(self.buffer) < self.batch_size:

            return None

        # Sample replay buffer

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            masks
        ) = self.buffer.sample(
            self.batch_size
        )

        # CURRENT Q VALUE

        q_values = self.q_network(
            states
        )

        # Pick Q(s,a) for the actions actually taken

        current_q = q_values.gather(
            1,
            actions.unsqueeze(1)
        ).squeeze(1)

        # TARGET Q VALUE

        with torch.no_grad():

            next_q_values = self.target_network(next_states)

            next_q_values = torch.where(
                masks > 0,
                next_q_values,
                torch.tensor(-float("inf"), device=self.device)
            )

            next_q = next_q_values.max(dim=1)[0]

            target_q = rewards.clone()

            non_terminal = dones == 0

            target_q[non_terminal] += (
                self.gamma * next_q[non_terminal]
            )

        # LOSS

        loss = torch.mean(
            (current_q - target_q) ** 2
        )

        # BACKPROPAGATION

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        # UPDATE STEP COUNT

        self.steps += 1

        # UPDATE TARGET NETWORK

        if self.steps % self.target_update_freq == 0:

            self.update_target_network()

        # EPSILON DECAY

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )

        # STORE LOSS

        self.losses.append(
            loss.item()
        )

        return loss.item()

    # UPDATE TARGET NETWORK

    def update_target_network(self):

        self.target_network.load_state_dict(
            self.q_network.state_dict()
        )

    # SAVE

    def save(self, filepath):

        torch.save(
            {
                "q_network":
                    self.q_network.state_dict(),

                "target_network":
                    self.target_network.state_dict(),

                "optimizer":
                    self.optimizer.state_dict(),

                "epsilon":
                    self.epsilon,

                "steps":
                    self.steps,

                "episodes":
                    self.episodes
            },
            filepath
        )

    # LOAD

    def load(self, filepath):

        checkpoint = torch.load(
            filepath,
            map_location=self.device
        )

        self.q_network.load_state_dict(
            checkpoint["q_network"]
        )

        self.target_network.load_state_dict(
            checkpoint["target_network"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        self.epsilon = checkpoint["epsilon"]

        self.steps = checkpoint["steps"]

        self.episodes = checkpoint["episodes"]

    # STATS

    def get_stats(self):

        return {
            "steps": self.steps,
            "episodes": self.episodes,
            "epsilon": self.epsilon,
            "buffer_size": len(self.buffer),
            "average_loss":
                np.mean(self.losses[-100:])
                if self.losses
                else 0.0
        }