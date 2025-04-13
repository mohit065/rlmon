import numpy as np
import asyncio
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player import Player

class PokemonEnvironment:
    def __init__(self, opponent: Player = None):
        self.opponent = opponent
        self.player = None
        
        self.observation_dim = 126
        self.action_dim = 9
        
        self.POKEMON_TYPES = ["normal", "fire", "water", "electric", "grass", "ice", "fighting","poison", 
            "ground", "flying", "psychic", "bug", "rock", "ghost","dragon", "dark", "steel"]
        
        self.TYPE_TO_IDX = {type: idx for idx, type in enumerate(self.POKEMON_TYPES)}
        self.STATUS_CONDITIONS = ["psn", "tox", "par", "slp", "frz", "brn"]
        
        self.STATUS_TO_IDX = {status: idx for idx, status in enumerate(self.STATUS_CONDITIONS)}
        
        self.current_battle = None
        self.done = False

    def set_player(self, player):
        self.player = player

    async def reset(self):
        if not self.player or not self.opponent:
            raise ValueError("Player and opponent must be set before calling reset")
        
        battle = await self.player.battle_against(self.opponent, n_battles=1)
        self.current_battle = battle
        self.done = False
        
        return battle

    async def step(self, action):
        if not self.current_battle:
            raise ValueError("Must call reset before step")
        
        await self.player._handle_action(action, self.current_battle.battle_tag)
        await asyncio.sleep(0.1)

        reward = self.compute_reward(self.current_battle)
        self.done = self.current_battle.ended
        
        info = {
            "battle_won": self.current_battle.won if self.current_battle.ended else None,
            "active_pokemon": self.current_battle.active_pokemon.species if self.current_battle.active_pokemon else None,
            "opponent_pokemon": self.current_battle.opponent_active_pokemon.species if self.current_battle.opponent_active_pokemon else None
        }
        
        return self.current_battle, reward, self.done, info

    def compute_reward(self, battle: AbstractBattle) -> float:
        return self.reward_computing_helper(
            battle,
            fainted_value=2.0,
            hp_value=1.0,
            victory_value=30.0
        )

    def reward_computing_helper(self,battle: AbstractBattle,fainted_value: float = 0.0,
        hp_value: float = 0.0,victory_value: float = 1.0) -> float:

        if battle.ended:
            return victory_value * (1 if battle.won else -1)
        
        total_hp_reward = 0

        if battle.active_pokemon:
            total_hp_reward += hp_value * battle.active_pokemon.current_hp_fraction

        if battle.opponent_active_pokemon:
            total_hp_reward -= hp_value * battle.opponent_active_pokemon.current_hp_fraction

        fainted_opponent = len([p for p in battle.opponent_team.values() if p.fainted])
        fainted_self = len([p for p in battle.team.values() if p.fainted])
        fainted_reward = fainted_value * (fainted_opponent - fainted_self)
        
        return total_hp_reward + fainted_reward

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        embedding = np.zeros(self.observation_dim, dtype=np.float32)

        if battle.active_pokemon:
            for type in battle.active_pokemon.types:
                if type and type in self.TYPE_TO_IDX:
                    embedding[self.TYPE_TO_IDX[type]] = 1.0

        if battle.opponent_active_pokemon:
            for type in battle.opponent_active_pokemon.types:
                if type and type in self.TYPE_TO_IDX:
                    embedding[18 + self.TYPE_TO_IDX[type]] = 1.0

        if battle.active_pokemon:
            embedding[36] = battle.active_pokemon.current_hp_fraction

        if battle.opponent_active_pokemon:
            embedding[37] = battle.opponent_active_pokemon.current_hp_fraction

        if battle.active_pokemon:
            if battle.active_pokemon.status:
                status = battle.active_pokemon.status.name.lower()
                if status in self.STATUS_TO_IDX:
                    embedding[118 + self.STATUS_TO_IDX[status]] = 1.0

        if battle.opponent_active_pokemon:
            if battle.opponent_active_pokemon.status:
                status = battle.opponent_active_pokemon.status.name.lower()
                if status in self.STATUS_TO_IDX:
                    embedding[126 + self.STATUS_TO_IDX[status]] = 1.0

        if battle.active_pokemon:
            for i, move in enumerate(battle.available_moves):
                if i < 4:
                    move_base = 38 + i * 20
                    if move.base_power:
                        embedding[move_base] = min(move.base_power / 100.0, 1.0)
                    
                    embedding[move_base + 1] = move.accuracy / 100.0 if move.accuracy else 1.0
                    
                    if move.type and move.type.name.lower() in self.TYPE_TO_IDX:
                        type_idx = self.TYPE_TO_IDX[move.type.name.lower()]
                        embedding[move_base + 2 + type_idx] = 1.0

        return embedding

    def action_space_size(self):
        return self.action_dim

    def observation_space_size(self):
        return self.observation_dim

    def action_to_move(self, battle, action_idx):
        available_moves = battle.available_moves
        available_switches = battle.available_switches

        if action_idx < 4:
            if action_idx < len(available_moves):
                return available_moves[action_idx]
            
            if available_moves:
                print("[Warning] Invalid move index, falling back to random move.")
                return np.random.choice(available_moves)
            
            return None
        else:
            switch_idx = action_idx - 4

            if switch_idx < len(available_switches):
                return available_switches[switch_idx]
            
            if available_switches:
                print("[Warning] Invalid switch index, falling back to random switch.")
                return np.random.choice(available_switches)
            
            return np.random.choice(available_moves) if available_moves else None
