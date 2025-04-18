import os
import time
import asyncio

from teams import teams
from agent import DQNAgent
from poke_env.player import RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer

LOG_FREQ = 10
SAVE_FREQ = 1000
NUM_EPISODES = 10000
MODEL_PATH = "model.pth"
BATTLE_FORMAT = "gen4anythinggoes"

async def main():
    agent_team_id = 0
    opponent_team_id = 1

    random_player = RandomPlayer(battle_format=BATTLE_FORMAT, team=teams[opponent_team_id])
    max_base_power_player = MaxBasePowerPlayer(battle_format=BATTLE_FORMAT, team=teams[opponent_team_id])
    simple_heuristics_player = SimpleHeuristicsPlayer(battle_format=BATTLE_FORMAT, team=teams[opponent_team_id])

    agent = DQNAgent(battle_format=BATTLE_FORMAT, team=teams[agent_team_id])
    opponent = random_player

    if os.path.exists(MODEL_PATH):
        agent.load_model(path=MODEL_PATH)
        print(f"LOADED EXISTING MODEL FROM {MODEL_PATH}. RESUMING TRAINING.")
    else:
        print(f"NO EXISTING MODEL FOUND AT {MODEL_PATH}. STARTING TRAINING FROM SCRATCH.")

    print(f"TRAINING: DQNAgent WITH TEAM {1+agent_team_id} VS {type(opponent).__name__} WITH TEAM {1+opponent_team_id}")
    print(f"STARTING TRAINING LOOP FOR {NUM_EPISODES} EPISODES...\n")

    start_time = time.time()
    episode_wins = []
    previous_total_wins = 0

    for episode in range(1, NUM_EPISODES + 1):
        await agent.battle_against(opponent, n_battles=1)

        current_total_wins = agent.n_won_battles
        last_battle_won = 1 if current_total_wins > previous_total_wins else 0
        episode_wins.append(last_battle_won)
        previous_total_wins = current_total_wins

        total_battles = agent.n_finished_battles
        total_win_rate = current_total_wins / total_battles if total_battles > 0 else 0.0
        total_steps = agent.steps_done

        if episode % LOG_FREQ == 0:
            elapsed_time = time.time() - start_time
            print(f"EP: {episode}/{NUM_EPISODES} | STEPS: {total_steps} | "
                f"WINRATE: {total_win_rate:.3f} ({current_total_wins}/{total_battles}) | "
                f"TIME: {elapsed_time:.2f}s")

        if episode % SAVE_FREQ == 0:
            agent.save_model(path=MODEL_PATH)
            print(f"\nSAVE {episode//SAVE_FREQ}: MODEL SAVED AT {MODEL_PATH}.\n")

    agent.save_model(path=MODEL_PATH)
    print(f"\nTRAINING LOOP FINISHED: {episode} EPISODES COMPLETED.")

    final_total_battles = agent.n_finished_battles
    final_total_wins = agent.n_won_battles
    final_win_rate = final_total_wins / final_total_battles if final_total_battles > 0 else 0.0
    print(f"FINAL W/R: {final_win_rate:.3f} ({final_total_wins}/{final_total_battles})")
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
    except KeyboardInterrupt:
        print("\nINTERRUPTED.")
    finally:
        print("TRAINING COMPLETED.")
