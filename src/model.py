import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Activation, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE
import os
from typing import List, Tuple, Dict
import random
from collections import deque

from poke_env.player import Player
from poke_env.environment import AbstractBattle

from spaces import PokemonEnvironment

class DQNAgent(Player):
    def __init__(self, 
                 battle_format: str,
                 player_configuration=None,
                 server_configuration=None,
                 input_dim: int = 126,
                 output_dim: int = 9,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.1,
                 epsilon_decay: float = 0.999,
                 gamma: float = 0.99,
                 learning_rate: float = 0.001,
                 memory_size: int = 10000,
                 batch_size: int = 32,
                 model_path: str = "models/pokemon_dqn.h5"):
        """
        DQN Agent for Pokemon battling using TensorFlow/Keras
        
        Args:
            battle_format: The battle format to use
            player_configuration: Configuration for the player
            server_configuration: Configuration for the server
            input_dim: Dimension of the input (observation space)
            output_dim: Dimension of the output (action space)
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon
            gamma: Discount factor
            learning_rate: Learning rate for the optimizer
            memory_size: Size of replay memory
            batch_size: Batch size for training
            model_path: Path to save/load the model
        """
        super().__init__(
            battle_format=battle_format,
            player_configuration=player_configuration,
            server_configuration=server_configuration
        )
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Create environment wrapper for embedding battles
        self.env = PokemonEnvironment(None)
        
        # Exploration parameters
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Discount factor
        self.gamma = gamma
        
        # Replay memory
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        
        # Setup model
        self.model = self._build_model(input_dim, output_dim, learning_rate)
        self.target_model = self._build_model(input_dim, output_dim, learning_rate)
        self.update_target_model()
        self.target_update_frequency = 1000
        self.update_counter = 0
        
        # Save path
        self.model_path = model_path
        
    def _build_model(self, input_dim, output_dim, learning_rate):
        """
        Build the neural network model
        
        Args:
            input_dim: Dimension of the input (observation space)
            output_dim: Dimension of the output (action space)
            learning_rate: Learning rate for the optimizer
            
        Returns:
            The built model
        """
        model = Sequential()
        model.add(Dense(256, input_dim=input_dim))
        model.add(Activation('relu'))
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(output_dim, activation='linear'))
        
        model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
        
        return model
    
    def update_target_model(self):
        """
        Update the target model with the weights of the main model
        """
        self.target_model.set_weights(self.model.get_weights())
    
    def choose_move(self, battle):
        """
        Choose a move for the current battle
        
        Args:
            battle: Current battle state
            
        Returns:
            Action to take
        """
        # Convert battle to state vector
        state = self.env.embed_battle(battle)
        state = np.reshape(state, [1, self.input_dim])
        
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            # Random action
            return self._action_to_move(battle, random.randrange(self.output_dim))
        else:
            # Greedy action
            q_values = self.model.predict(state, verbose=0)
            
            # Get the action with highest Q-value
            action = np.argmax(q_values[0])
            return self._action_to_move(battle, action)
    
    def _action_to_move(self, battle, action_idx):
        """
        Convert action index to actual move or switch
        
        Args:
            battle: Current battle
            action_idx: Index of the action
            
        Returns:
            The selected move or switch
        """
        return self.env.action_to_move(battle, action_idx)
    
    def remember(self, state, action, reward, next_state, done):
        """
        Add experience tuple to replay memory
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """
        Train the network using experience replay
        """
        if len(self.memory) < self.batch_size:
            return
        
        # Sample mini-batch from memory
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Arrays to store batch data
        states = np.zeros((self.batch_size, self.input_dim))
        next_states = np.zeros((self.batch_size, self.input_dim))
        actions, rewards, dones = [], [], []
        
        # Fill arrays with data from experiences
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state
            next_states[i] = next_state
            actions.append(action)
            rewards.append(reward)
            dones.append(float(done))
        
        # Convert to numpy arrays
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)
        
        # Predict Q-values for current states
        targets = self.model.predict(states, verbose=0)
        
        # Predict Q-values for next states using target model
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        # Update targets for actions taken
        for i in range(self.batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train model on batch
        self.model.fit(states, targets, epochs=1, verbose=0)
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        # Update target model periodically
        self.update_counter += 1
        if self.update_counter % self.target_update_frequency == 0:
            self.update_target_model()
    
    def save_model(self, path=None):
        """
        Save the model to disk
        
        Args:
            path: Path to save the model (if None, use default)
        """
        if path is None:
            path = self.model_path
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path=None):
        """
        Load the model from disk
        
        Args:
            path: Path to load the model from (if None, use default)
        """
        if path is None:
            path = self.model_path
        
        if os.path.exists(path):
            self.model = load_model(path)
            self.update_target_model()
            print(f"Model loaded from {path}")
        else:
            print(f"No model found at {path}")