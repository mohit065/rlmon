"""Environment setup for Pokémon battles."""
import asyncio
from poke_env.player.random_player import RandomPlayer
from agent1 import Agent1
from agent2 import Agent2
from config import get_battle_format

class BattleEnvironment:
    """Battle environment for two agents."""
    
    def __init__(self):
        self.battle_format = get_battle_format()
        self.agent1 = Agent1(self.battle_format)
        self.agent2 = Agent2(self.battle_format)
    
    async def run_battle(self, n_battles=1):
        """Run battles between the two agents."""
        # Start both agents
        await self.agent1.battle_against(self.agent2, n_battles=n_battles)
        
        # Calculate and print win rates
        print(f"Agent 1 win rate: {self.agent1.n_won_battles / n_battles}")
        print(f"Agent 2 win rate: {self.agent2.n_won_battles / n_battles}")
        
        # Debugging information
        print(f"Number of battles completed: {self.agent1.n_finished_battles}")
        
        # Close connections
        await self.agent1.close_connections()
        await self.agent2.close_connections()