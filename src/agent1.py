from poke_env.player import RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer
from poke_env.player import Player
from poke_env import AccountConfiguration
from poke_env import LocalhostServerConfiguration
from config import get_account_config, get_server_config

class Agent1(RandomPlayer):
    def __init__(self, battle_format):
        account_config = get_account_config("account1")
        server_config = get_server_config()
        account_configuration = AccountConfiguration(
            username = account_config["username"],
            password = account_config["password"]
        )
        super().__init__(
            account_configuration = account_configuration,
            battle_format = battle_format
        )