import asyncio
import logging
import webbrowser
from agent import Agent
from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer

agent_credentials = ("rlmonbot", "rlmonbot")
tester_credentials = ("rlmon2", "rlmon2")
battle_format = "gen4randombattle"

# Configure logging
logging.getLogger("poke_env").setLevel(logging.ERROR)

# Define the Tester class
class Tester(RandomPlayer):
    def __init__(self, username, password, battle_format):
        account_configuration = AccountConfiguration(username, password)
        super().__init__(account_configuration=account_configuration, battle_format=battle_format)

# Define the BattleEnvironment class
class BattleEnvironment:
    def __init__(self, agent_credentials, tester_credentials, battle_format):
        self.battle_format = battle_format

        agent_username, agent_password = agent_credentials
        tester_username, tester_password = tester_credentials

        # Initialize the agent and tester
        self.agent = Agent(agent_username, agent_password, self.battle_format)
        self.tester = Tester(tester_username, tester_password, self.battle_format)

    async def run_battle(self, n_battles=1):
        """Run a number of battles between agent and tester."""
        await self.tester.battle_against(self.agent, n_battles=n_battles)

        print(f"Battles completed: {self.tester.n_finished_battles}")
        print(f"Tester win rate: {self.tester.n_won_battles / n_battles}")
        print(f"Agent win rate: {self.agent.n_won_battles / n_battles}")

# Function to open the Showdown UI in a web browser
def open_showdown_ui(host="localhost", port=8000):
    url = f"http://{host}:{port}"
    print(f"Opening Showdown at {url}")
    webbrowser.open(url)

# Main asynchronous function to set up and run battles
async def main():
    print("Setting up battle environment...")
    environment = BattleEnvironment(agent_credentials, tester_credentials, battle_format)

    open_showdown_ui()
    await asyncio.sleep(2)

    print("Starting battles...")
    await environment.run_battle(n_battles=10)
    print("Battles completed!")

# Run the app
if __name__ == "__main__":
    print("Starting...")
    asyncio.get_event_loop().run_until_complete(main())