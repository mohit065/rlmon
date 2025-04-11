import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player import Player


class PokemonEnvironment:
    def __init__(self, opponent: Player, battle_format: str = "gen8randombattle"):
        """
        Custom Pokemon environment that doesn't use OpenAI Gym.
        
        Args:
            opponent: The opponent to battle against
            battle_format: The battle format to use
        """
        self.opponent = opponent
        self.battle_format = battle_format
        
        # Define observation and action space dimensions
        self.observation_dim = 126
        self.action_dim = 9
        
        # Map for pokemon types to indices
        self.POKEMON_TYPES = [
            "normal", "fire", "water", "electric", "grass", "ice", "fighting",
            "poison", "ground", "flying", "psychic", "bug", "rock", "ghost",
            "dragon", "dark", "steel", "fairy"
        ]
        
        self.TYPE_TO_IDX = {type: idx for idx, type in enumerate(self.POKEMON_TYPES)}
        
        # Map for status conditions
        self.STATUS_CONDITIONS = [
            "psn", "tox", "par", "slp", "frz", "brn", "confused", "trapped"
        ]
        
        self.STATUS_TO_IDX = {status: idx for idx, status in enumerate(self.STATUS_CONDITIONS)}
        
        # For tracking current battle
        self.current_battle = None
        self.done = False

    def reset(self, battle):
        """
        Reset the environment at the start of a new episode/battle.
        
        Args:
            battle: The new battle
            
        Returns:
            The initial observation
        """
        self.current_battle = battle
        self.done = False
        
        return self.embed_battle(battle)

    def step(self, battle, action):
        """
        Take a step in the environment.
        
        Args:
            battle: The current battle
            action: The action to take
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        self.current_battle = battle
        
        # Calculate reward
        reward = self.compute_reward(battle)
        
        # Check if done
        self.done = battle.ended
        
        # Get next state
        next_state = self.embed_battle(battle)
        
        # Info dictionary for additional info
        info = {
            "battle_won": battle.won if battle.ended else None,
            "active_pokemon": battle.active_pokemon.species if battle.active_pokemon else None,
            "opponent_pokemon": battle.opponent_active_pokemon.species if battle.opponent_active_pokemon else None
        }
        
        return next_state, reward, self.done, info

    def compute_reward(self, battle: AbstractBattle) -> float:
        """
        Returns the reward for the current state of the battle.
        
        Args:
            battle: The battle from which to compute the reward
            
        Returns:
            The reward as a float
        """
        return self.reward_computing_helper(
            battle,
            fainted_value=2.0,
            hp_value=1.0,
            victory_value=30.0
        )

    def reward_computing_helper(
        self,
        battle: AbstractBattle,
        fainted_value: float = 0.0,
        hp_value: float = 0.0,
        victory_value: float = 1.0
    ) -> float:
        """
        Computes the rewards from the given battle.
        
        Args:
            battle: The battle to compute rewards from
            fainted_value: The reward for fainted opponent Pokemon
            hp_value: The reward for remaining HP
            victory_value: The reward for winning the battle
            
        Returns:
            The computed reward
        """
        if battle.ended:
            return victory_value * (1 if battle.won else -1)
            
        # Calculate raw HP reward
        total_hp_reward = 0
        
        # Reward for our active Pokemon's health
        if battle.active_pokemon:
            total_hp_reward += hp_value * battle.active_pokemon.current_hp_fraction
        
        # Penalty for opponent's active Pokemon's health
        if battle.opponent_active_pokemon:
            total_hp_reward -= hp_value * battle.opponent_active_pokemon.current_hp_fraction
            
        # Reward for fainted opponent Pokemon
        fainted_opponent = len([p for p in battle.opponent_team.values() if p.fainted])
        fainted_self = len([p for p in battle.team.values() if p.fainted])
        
        # Calculate fainted reward
        fainted_reward = fainted_value * (fainted_opponent - fainted_self)
        
        return total_hp_reward + fainted_reward

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        """
        Converts a battle to a vector representation for the neural network.
        
        Args:
            battle: The battle to embed
            
        Returns:
            A vector representation of the battle
        """
        # Initialize the embedding vector
        embedding = np.zeros(self.observation_dim, dtype=np.float32)
        
        # 1. Active Pokemon types (one-hot encoding)
        if battle.active_pokemon:
            for type in battle.active_pokemon.types:
                if type and type in self.TYPE_TO_IDX:
                    embedding[self.TYPE_TO_IDX[type]] = 1.0
                    
        # 2. Opponent's active Pokemon types (one-hot encoding)
        if battle.opponent_active_pokemon:
            for type in battle.opponent_active_pokemon.types:
                if type and type in self.TYPE_TO_IDX:
                    embedding[18 + self.TYPE_TO_IDX[type]] = 1.0
                    
        # 3. Active Pokemon HP percentage
        if battle.active_pokemon:
            embedding[36] = battle.active_pokemon.current_hp_fraction
            
        # 4. Opponent's active Pokemon HP percentage
        if battle.opponent_active_pokemon:
            embedding[37] = battle.opponent_active_pokemon.current_hp_fraction
            
        # 5. Status conditions of active pokemon
        if battle.active_pokemon:
            if battle.active_pokemon.status:
                status = battle.active_pokemon.status.name.lower()
                if status in self.STATUS_TO_IDX:
                    embedding[118 + self.STATUS_TO_IDX[status]] = 1.0
            if battle.active_pokemon.is_confused():
                embedding[118 + self.STATUS_TO_IDX["confused"]] = 1.0
            if battle.active_pokemon.trapped:
                embedding[118 + self.STATUS_TO_IDX["trapped"]] = 1.0
                
        # 6. Status conditions of opponent active pokemon
        if battle.opponent_active_pokemon:
            if battle.opponent_active_pokemon.status:
                status = battle.opponent_active_pokemon.status.name.lower()
                if status in self.STATUS_TO_IDX:
                    embedding[118 + 4 + self.STATUS_TO_IDX[status]] = 1.0
            if battle.opponent_active_pokemon.is_confused():
                embedding[118 + 4 + self.STATUS_TO_IDX["confused"]] = 1.0
            if battle.opponent_active_pokemon.trapped:
                embedding[118 + 4 + self.STATUS_TO_IDX["trapped"]] = 1.0
                
        # 7. Moves information (base power, accuracy, type)
        # We'll only include available moves
        if battle.active_pokemon:
            for i, move in enumerate(battle.available_moves):
                if i < 4:  # Limit to 4 moves
                    move_base = 38 + i * 20  # Each move uses 20 spots (1 power + 1 accuracy + 18 types)
                    
                    # Base power (normalized to [-1, 1])
                    if move.base_power:
                        embedding[move_base] = min(move.base_power / 100.0, 1.0)
                    
                    # Accuracy (normalized to [0, 1])
                    if move.accuracy:
                        embedding[move_base + 1] = move.accuracy / 100.0
                    else:
                        embedding[move_base + 1] = 1.0  # Moves that never miss
                    
                    # Move type (one-hot encoding)
                    if move.type and move.type.name.lower() in self.TYPE_TO_IDX:
                        type_idx = self.TYPE_TO_IDX[move.type.name.lower()]
                        embedding[move_base + 2 + type_idx] = 1.0
        
        return embedding

    def action_space_size(self):
        """
        Returns the size of the action space
        
        Returns:
            The size of the action space
        """
        return self.action_dim

    def observation_space_size(self):
        """
        Returns the size of the observation space
        
        Returns:
            The size of the observation space
        """
        return self.observation_dim

    def action_to_move(self, battle, action_idx):
        """
        Convert action index to actual move or switch
        
        Args:
            battle: Current battle
            action_idx: Index of the action
            
        Returns:
            The selected move or switch
        """
        # If action is 0-3, it's a move; if 4-8, it's a switch
        available_moves = battle.available_moves
        available_switches = battle.available_switches
        
        if action_idx < 4:  # Move
            if action_idx < len(available_moves):
                return available_moves[action_idx]
            # If the move is not available, choose a random available move
            return np.random.choice(available_moves) if available_moves else None
        else:  # Switch (action_idx - 4 gives the index of the switch)
            switch_idx = action_idx - 4
            if switch_idx < len(available_switches):
                return available_switches[switch_idx]
            # If the switch is not available, choose a random available switch or move
            if available_switches:
                return np.random.choice(available_switches)
            return np.random.choice(available_moves) if available_moves else None