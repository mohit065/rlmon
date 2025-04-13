from poke_env.player import MaxBasePowerPlayer
from poke_env import AccountConfiguration

class Agent(MaxBasePowerPlayer):
    def __init__(self, username, password, battle_format):
        account_configuration = AccountConfiguration(username, password)
        super().__init__(account_configuration=account_configuration, battle_format=battle_format)