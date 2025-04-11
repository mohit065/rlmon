# from poke_env.player import RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer
# from poke_env.player import Player
# from poke_env import AccountConfiguration
# from poke_env import LocalhostServerConfiguration
# from config import get_account_config, get_server_config

# class Agent2(MaxBasePowerPlayer):
#     def __init__(self, battle_format):
#         account_config = get_account_config("account2")
#         server_config = get_server_config()
#         account_configuration = AccountConfiguration(
#             username = account_config["username"],
#             password = account_config["password"]
#         )
#         super().__init__(
#             account_configuration = account_configuration,
#             battle_format = battle_format
#         )

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

from poke_env.player import Player
from poke_env import AccountConfiguration
from poke_env import ServerConfiguration
from config import get_account_config, get_server_config
from spaces import PokemonEnvironment

class RLAgent(Player):
    def __init__(self, battle_format, model_path="models/pokemon_dqn.h5"):
        """
        RL Agent using a trained DQN model with TensorFlow/Keras
        
        Args:
            battle_format: Format of the battles
            model_path: Path to the trained model
        """
        # Set up configuration
        account_config = get_account_config("account2")
        server_config = get_server_config()
        
        account_configuration = AccountConfiguration(
            username=account_config["username"],
            password=account_config["password"]
        )
        
        server_configuration = ServerConfiguration(
            **server_config
        )
        
        super().__init__(
            account_configuration=account_configuration,
            server_configuration=server_configuration,
            battle_format=battle_format
        )
        
        self.model_path = model_path
        
        # Set up model dimensions
        self.input_dim = 126  # Same as the one used for training
        self.output_dim = 9   # 4 moves + 5 switches
        
        # Load model if it exists
        self.model = None
        self.load_model()
        
        # For embedding battles
        self.env = PokemonEnvironment(None)
    
    def load_model(self):
        """
        Load the trained model
        """
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
            print(f"Model loaded from {self.model_path}")
        else:
            print(f"No model found at {self.model_path}. Using random actions.")
    
    def choose_move(self, battle):
        """
        Choose a move for the current battle
        
        Args:
            battle: Current battle state
        
        Returns:
            Action to take
        """
        # If no model loaded, use random actions
        if self.model is None:
            return self.choose_random_move(battle)
        
        # Convert battle to state vector
        state = self.env.embed_battle(battle)
        state = np.reshape(state, [1, self.input_dim])
        
        # Get Q-values from the model
        q_values = self.model.predict(state, verbose=0)[0]
        
        # Get action indices sorted by Q-value
        action_indices = np.argsort(q_values)[::-1]
        
        # Try each action in order of Q-value until we find a valid one
        for action_idx in action_indices:
            if action_idx < 4:  # Move
                # Check if move is available
                if action_idx < len(battle.available_moves):
                    return battle.available_moves[action_idx]
            else:  # Switch
                switch_idx = action_idx - 4
                # Check if switch is available
                if switch_idx < len(battle.available_switches):
                    return battle.available_switches[switch_idx]
        
        # If no valid action found, choose randomly
        return self.choose_random_move(battle)
    
    def choose_random_move(self, battle):
        """
        Choose a random move or switch
        
        Args:
            battle: Current battle
        
        Returns:
            Random action
        """
        # Combine available moves and switches
        available_moves = battle.available_moves
        available_switches = battle.available_switches
        
        all_actions = available_moves + available_switches
        
        # Choose random action
        if all_actions:
            return np.random.choice(all_actions)
        
        # If no actions available, return None and let the environment handle it
        return None

# This is the agent that will be used in battles
class Agent2(RLAgent):
    def __init__(self, battle_format):
        super().__init__(battle_format=battle_format)