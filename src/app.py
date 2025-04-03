"""Main application for running Pokémon battles."""
import asyncio
import os
import sys
import webbrowser
from environment import BattleEnvironment
from config import get_server_config

def open_showdown_ui():
    """Open the Pokémon Showdown UI in a web browser."""
    server_config = get_server_config()
    url = f"http://{server_config['host']}:{server_config['port']}"
    print(f"Opening Pokémon Showdown UI at {url}")
    webbrowser.open(url)

async def main():
    """Main function to run the battles."""
    print("Setting up battle environment...")
    environment = BattleEnvironment()
    
    # Open the Showdown UI before starting battles
    open_showdown_ui()
    
    # Wait a moment for the UI to load
    await asyncio.sleep(2)
    
    print("Starting battles...")
    await environment.run_battle(n_battles=5)
    print("Battles completed!")

if __name__ == "__main__":
    print("Starting Pokémon Showdown Battle Bot...")
    asyncio.get_event_loop().run_until_complete(main())