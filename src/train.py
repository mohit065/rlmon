import time
import asyncio
import logging

from agent import DQNAgent
from poke_env.player import RandomPlayer, MaxBasePowerPlayer

BATTLE_FORMAT = "gen4randombattle"
NUM_EPISODES = 10000
LOG_FREQ = 10
SAVE_FREQ = 1000

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
opponent = RandomPlayer(battle_format=BATTLE_FORMAT) # needs to be here

async def train_dqn(agent, opponent):
    start_time = time.time()
    win_rates = []
    total_steps = agent.steps_done
    print(f"\nStarting training for {NUM_EPISODES} episodes against {opponent.__class__.__name__}...")

    for episode in range(1, NUM_EPISODES + 1):
        agent.last_battle_state = None
        agent.last_action_idx = None
        try:
            await agent.battle_against(opponent, n_battles=1)
            current_wins = agent.n_won_battles
            total_battles = agent.n_finished_battles
            win_rate = current_wins / total_battles if total_battles > 0 else 0
            win_rates.append(win_rate)
            total_steps = agent.steps_done

            if episode % LOG_FREQ == 0:
                elapsed_time = time.time() - start_time
                print(f"Episode: {episode}/{NUM_EPISODES} | "
                      f"Steps: {total_steps} | "
                      f"Total Win Rate: {win_rate:.3f} ({current_wins}/{total_battles}) | "
                      f"Time: {elapsed_time:.2f}s")

            if episode % SAVE_FREQ == 0:
                agent.save_model()

        except Exception as e:
            print(f"Error during episode {episode}: {e}")
            agent.reset_battles()
            await asyncio.sleep(1)

    agent.save_model()
    print(f"\nTraining finished after {NUM_EPISODES} episodes.")
    print(f"Final Win Rate: {agent.n_won_battles / agent.n_finished_battles:.3f}")
    print(f"Total Steps: {total_steps}")


async def main():
    print("Setting up DQN agent and opponent...")
    player = DQNAgent(battle_format=BATTLE_FORMAT) # needs to be here as well
    opponent = RandomPlayer(battle_format=BATTLE_FORMAT) # also needs to be here
    await train_dqn(agent=player,opponent=opponent)

if __name__ == "__main__":
    print("Starting DQN Training...")
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(main())
        else:
            loop.run_until_complete(main())

    except RuntimeError:
        asyncio.run(main())

    print("Training script finished.")