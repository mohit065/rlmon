import os
import asyncio

from teams import teams
from agent import DQNAgent
from poke_env.player import RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer

# Config
agent_team_id = 0
opponent_team_id = 0
opponent_class = RandomPlayer

NUM_EPISODES = 100
BATTLE_FORMAT = "gen4anythinggoes"
MODEL_DIR = "../models"

async def evaluate():
    # Set agent and opponent
    opponent = opponent_class(battle_format=BATTLE_FORMAT, team=teams[opponent_team_id])
    agent = DQNAgent(battle_format=BATTLE_FORMAT, team=teams[agent_team_id])
    model_path = f"{MODEL_DIR}/{opponent.__class__.__name__}_{1+agent_team_id}_{1+opponent_team_id}.pth"

    if not os.path.exists(model_path):
        print(f"NO MODEL FOUND TO EVALUATE AT {model_path}")
        return

    agent.load_model(model_path)
    print(f"\nEVALUATING: DQNAgent (TEAM {1+agent_team_id}) VS {opponent.__class__.__name__} (TEAM {1+opponent_team_id})")
    print(f"USING MODEL: {model_path}")
    print(f"RUNNING {NUM_EPISODES} EPISODES...\n")

    # Evaluate
    await agent.battle_against(opponent, n_battles=NUM_EPISODES)

    # Log results
    win_rate = agent.n_won_battles / agent.n_finished_battles if agent.n_finished_battles > 0 else 0.0
    print(f"FINAL WIN RATE: {win_rate:.3f} ({agent.n_won_battles}/{agent.n_finished_battles})")

if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(evaluate())
        else:
            loop.run_until_complete(evaluate())
    except RuntimeError as e:
        asyncio.run(evaluate())
    except KeyboardInterrupt as e:
        print("INTERRUPTED")
    finally:
        print("ENDING...")