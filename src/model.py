import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from poke_env.player import Player
from spaces import PokemonEnvironment

class DQNAgent(Player):
    def __init__(self, 
                 input_dim: int = 126,
                 output_dim: int = 9,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.1,
                 epsilon_decay: float = 0.995,
                 gamma: float = 0.99,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 model_path: str = "pokemon_dqn.pth"):
        super().__init__()
        
        # Agent parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.batch_size = batch_size
        self.model_path = model_path
        
        # Initialize environment
        self.env = PokemonEnvironment(None)
        
        # Initialize the memory (simple list to hold the experience replay)
        self.memory = []
        self.model = self.build_model(input_dim, output_dim)
        self.target_model = self.build_model(input_dim, output_dim)
        self.update_target_model()
        
        # Target update frequency
        self.target_update_frequency = 500
        self.update_counter = 0

    def build_model(self, input_dim, output_dim):
        """Builds the neural network model for the agent."""
        model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        return model
    
    def update_target_model(self):
        """Copies the weights from the model to the target model."""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def choose_move(self, battle):
        """Choose a move based on the exploration-exploitation tradeoff."""
        state = self.env.embed_battle(battle)
        state = torch.tensor(np.reshape(state, [1, self.input_dim]), dtype=torch.float32)
        if np.random.random() < self.epsilon:
            return self.action_to_move(battle, random.randrange(self.output_dim))
        else:
            with torch.no_grad():
                q_values = self.model(state)
            action = torch.argmax(q_values[0]).item()
            return self.action_to_move(battle, action)
    
    def action_to_move(self, battle, action_idx):
        """Converts the action index to a move."""
        return self.env.action_to_move(battle, action_idx)
    
    def remember(self, state, action, reward, next_state, done):
        """Stores the experience in the memory."""
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 10000:  # Keep the memory size manageable
            self.memory.pop(0)  # Removes the oldest experience if memory is full
    
    def replay(self):
        """Trains the model using a sample of the experiences stored in memory."""
        if len(self.memory) < self.batch_size:
            return  # Not enough memory to sample
        
        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.zeros((self.batch_size, self.input_dim))
        next_states = torch.zeros((self.batch_size, self.input_dim))
        actions, rewards, dones = [], [], []
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = torch.tensor(state, dtype=torch.float32)
            next_states[i] = torch.tensor(next_state, dtype=torch.float32)
            actions.append(action)
            rewards.append(reward)
            dones.append(float(done))

        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)
        dones = torch.tensor(dones)
        
        q_values = self.model(states)
        next_q_values = self.target_model(next_states)
        
        targets = q_values.clone()
        
        for i in range(self.batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * torch.max(next_q_values[i])
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Perform gradient descent
        optimizer.zero_grad()
        loss = criterion(q_values, targets)
        loss.backward()
        optimizer.step()
        
        # Decay epsilon after every replay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.update_counter += 1
        if self.update_counter % self.target_update_frequency == 0:
            self.update_target_model()
    
    def save_model(self, path=None):
        """Saves the current model to a file."""
        if path is None:
            path = self.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self, path=None):
        """Loads the model from a file."""
        if path is None:
            path = self.model_path
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
            self.update_target_model()
            print(f"Model loaded from {path}")
        else:
            print(f"No model found at {path}")