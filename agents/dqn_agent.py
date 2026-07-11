import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from env.game import PolytopiaEnv

class QNetwork(nn.Module):
    """
    Maps a state observation to Q values for each action.
 
    Input:  110 floats (fog map + position + last-action one-hot)
    Output: 8 floats (Q value for each action)
    """

    def __init__(self, obs_size: int, num_actions: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(obs_size,  128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x):
        return self.network(x)
    

class ReplayBuffer:
    """
    Stores past experiences and allows random sampling.
    Once full, oldest experiences are overwritten.
    """
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store one experience tuple."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """Randomly sample a batch of experiences"""
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)
    
        return (
            torch.tensor(np.array(states),      dtype=torch.float32),
            torch.tensor(np.array(actions),     dtype=torch.long),
            torch.tensor(np.array(rewards),     dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones),       dtype=torch.float32),
        )
    
    def __len__(self):
        return len(self.buffer)
    

class DQNAgent:
    """
    Deep Q-Network agent.
 
    Learns to explore the grid by training a neural network
    to predict Q values for each action.
    """

    def __init__(
        self,
        obs_size: int = 110,
        num_actions: int = 8,
        lr: float = 1e-4,           # learning rate
        gamma: float = 0.99,        # discount factor
        epsilon_start: float = 1.0, # initial exploration rate
        epsilon_end: float = 0.05,  # minimum exploration rate
        epsilon_decay: float = 0.999, # how fast epsilon decays
        buffer_capacity: int = 10000,
        batch_size: int = 64,
        train_start: int = 500,     # steps before training begins
    ):
        
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.train_start = train_start

        # neural network + optimizer
        self.q_network = QNetwork(obs_size, num_actions)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.target_network = QNetwork(obs_size, num_actions)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_update_freq = 100  # sync every 100 steps
        self.steps_done = 0
        self.loss_fn = nn.MSELoss()

        # replay buffer
        self.buffer = ReplayBuffer(buffer_capacity)

        #step counter
        self.steps_done = 0

    def select_action(self, obs, env=None) -> int:
        """
        Epsilon-greedy action selection.
        Explore randomly with probability epsilon,
        otherwise pick the action with the highest Q value.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        
        # convert obs to tensor and get Q values from network
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(obs_tensor)
        return q_values.argmax().item()

    def store_experience(self, state, action, reward, next_state, done):
        """Add one experience to the replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        """
        Sample a batch from the buffer and update the network.
        Returns the loss value, or None if training hasn't started yet.
        """
        
        if len(self.buffer) < self.train_start:
            return None
        
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # current Q values for the actions we took
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # target Q values using Bellman equation (Double DQN: select the best
        # next action with the online network, evaluate it with the target
        # network, to avoid the overestimation bias of vanilla DQN)
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            max_next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        # compute loss and update weights
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # sync target network periodically
        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()
    
    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        """Save the network weights to a file."""
        torch.save(self.q_network.state_dict(), path)

    def load(self, path: str):
        """Load network weights from a file."""
        self.q_network.load_state_dict(torch.load(path))

