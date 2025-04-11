import os
import sys
import asyncio
import logging
import webbrowser
from config import get_server_config
from environment import BattleEnvironment

logging.getLogger("poke_env.player.player").setLevel(logging.ERROR)

def open_showdown_ui():
    server_config = get_server_config()
    url = f"http://{server_config['host']}:{server_config['port']}"
    print(f"Opening Pokémon Showdown UI at {url}")
    webbrowser.open(url)

async def main():
    print("Setting up battle environment...")
    environment = BattleEnvironment()

    open_showdown_ui()
    await asyncio.sleep(2)
    
    print("Starting battles...")
    await environment.run_battle(n_battles=5)
    print("Battles completed!")

if __name__ == "__main__":
    print("Starting Pokémon Showdown Battle Bot...")
    asyncio.get_event_loop().run_until_complete(main())