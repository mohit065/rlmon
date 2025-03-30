"""Agent 1 implementation - can be configured as RandomPlayer or MaxDamagePlayer."""
from poke_env.player import RandomPlayer
from poke_env.player import Player
from poke_env import AccountConfiguration
from poke_env import LocalhostServerConfiguration
from config import get_account_config, get_server_config

class Agent1(RandomPlayer):
    """
    Agent 1 implementation using RandomPlayer strategy.
    This can be modified to use MaxDamagePlayer by changing the inheritance.
    """
    
    def __init__(self, battle_format):
        # Get configuration
        account_config = get_account_config("account1")
        server_config = get_server_config()
        
        # Set up account configuration
        account_configuration = AccountConfiguration(
            username=account_config["username"],
            password=account_config["password"]
        )
        
        # Initialize the player with default localhost configuration
        super().__init__(
            account_configuration=account_configuration,
            battle_format=battle_format
        )

# Uncomment and modify this to use MaxDamagePlayer instead
"""
from poke_env.player import MaxDamagePlayer

class Agent1(MaxDamagePlayer):
    # Same implementation as above, just change the parent class
    ...
"""