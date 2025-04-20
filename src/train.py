import time
import asyncio

from teams import teams
from agent import DQNAgent
from poke_env.player import RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer

LOG_FREQ = 50                           # Logging frequency
NUM_EPISODES = 5000                     # Number of episodes to train on
BATTLE_FORMAT = "gen4anythinggoes"      # Battle format

async def main():
    # Agent and Opponent teams
    agent_team_id = 0
    opponent_team_id = 1

    # 3 classes of opponents
    opponent1 = RandomPlayer(battle_format=BATTLE_FORMAT, team=teams[opponent_team_id])
    opponent2 = MaxBasePowerPlayer(battle_format=BATTLE_FORMAT, team=teams[opponent_team_id])
    opponent3 = SimpleHeuristicsPlayer(battle_format=BATTLE_FORMAT, team=teams[opponent_team_id])

    agent = DQNAgent(battle_format=BATTLE_FORMAT, team=teams[agent_team_id])
    opponent = opponent1

    print(f"TRAINING: DQNAgent (TEAM {1+agent_team_id}) VS {type(opponent).__name__} (TEAM {1+opponent_team_id})")
    print(f"STARTING TRAINING LOOP FOR {NUM_EPISODES} EPISODES...\n")

    start_time = time.time()

    # Training
    for episode in range(1, NUM_EPISODES + 1):
        # Perform a battle
        await agent.battle_against(opponent, n_battles=1)

        # Calculate metrics
        total_wins = agent.n_won_battles
        total_battles = agent.n_finished_battles
        total_win_rate = total_wins / total_battles if total_battles > 0 else 0.0
        total_steps = agent.steps_done

        # Log metrics
        if episode % LOG_FREQ == 0:
            elapsed_time = time.time() - start_time
            print(f"EP: {episode}/{NUM_EPISODES} | STEPS: {total_steps} | "
                f"WINRATE: {total_win_rate:.4f} ({total_wins}/{total_battles}) | "
                f"TIME: {elapsed_time:.2f}s")

    final_total_battles = agent.n_finished_battles
    final_total_wins = agent.n_won_battles
    final_win_rate = final_total_wins / final_total_battles if final_total_battles > 0 else 0.0
    print(f"\nFINAL W/R: {final_win_rate:.4f} ({final_total_wins}/{final_total_battles})")
    print(f"FINAL STEPS: {agent.steps_done}")

if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(main())
        else:
            loop.run_until_complete(main())

    except RuntimeError as e:
        asyncio.run(main())

    finally:
        print("ENDING...")