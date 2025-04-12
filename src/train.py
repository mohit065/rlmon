import asyncio
import time
import numpy as np
import os
import math

from poke_env.player import RandomPlayer, MaxBasePowerPlayer
from dqn_agent import DQNAgent
from config import get_battle_format, get_training_params, get_dqn_params

import logging

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# --- Opponent Selection ---
# Choose the opponent agent for training
# from agent1 import Agent1 # If you want to train against RandomPlayer defined in agent1.py
# opponent = Agent1(get_battle_format())
opponent = RandomPlayer(battle_format=get_battle_format())
# opponent = MaxBasePowerPlayer(battle_format=get_battle_format())


async def train_dqn(dqn_agent, opponent_player, num_episodes, log_freq, save_freq):
    start_time = time.time()
    win_rates = []
    total_steps = dqn_agent.steps_done # Continue counting steps if model loaded

    print(f"\nStarting training for {num_episodes} episodes against {opponent_player.__class__.__name__}...")

    for episode in range(1, num_episodes + 1):
        # Reset agent's internal state tracker for the new battle
        dqn_agent.last_battle_state = None
        dqn_agent.last_action_idx = None

        try:
            # Play one battle (episode)
            await dqn_agent.battle_against(opponent_player, n_battles=1)

            # Logging after the battle finishes
            current_wins = dqn_agent.n_won_battles
            total_battles = dqn_agent.n_finished_battles
            win_rate = current_wins / total_battles if total_battles > 0 else 0
            win_rates.append(win_rate)

            total_steps = dqn_agent.steps_done # Update total steps count

            if episode % log_freq == 0:
                avg_win_rate_last_log = np.mean(win_rates[-log_freq:])
                elapsed_time = time.time() - start_time
                print(f"Episode: {episode}/{num_episodes} | "
                      f"Steps: {total_steps} | "
                      f"Avg Win Rate ({log_freq} eps): {avg_win_rate_last_log:.3f} | "
                      f"Total Win Rate: {win_rate:.3f} ({current_wins}/{total_battles}) | "
                      f"Epsilon: {dqn_agent.epsilon_end + (dqn_agent.epsilon_start - dqn_agent.epsilon_end) * math.exp(-1. * total_steps / dqn_agent.epsilon_decay):.3f} | "
                      f"Time: {elapsed_time:.2f}s")

            # Save model periodically
            if episode % save_freq == 0:
                 dqn_agent.save_model()

        except Exception as e:
            print(f"Error during episode {episode}: {e}")
            # Consider stopping or adding more robust error handling
            # Reset agent state just in case
            dqn_agent.reset_battles()
            await asyncio.sleep(1) # Wait a bit before next battle attempt

    # Final save
    dqn_agent.save_model()
    print(f"\nTraining finished after {num_episodes} episodes.")
    print(f"Final Win Rate: {dqn_agent.n_won_battles / dqn_agent.n_finished_battles:.3f}")
    print(f"Total Steps: {total_steps}")


async def main():
    print("Setting up DQN agent and opponent...")
    battle_format = get_battle_format()
    training_params = get_training_params()
    dqn_params = get_dqn_params() # Needed by agent

    # Ensure model directory exists if needed
    model_dir = os.path.dirname(training_params['model_save_path'])
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Instantiate the DQN Agent in training mode
    player = DQNAgent(battle_format=battle_format, train_mode=True)

    # Instantiate the opponent
    opponent = RandomPlayer(battle_format=battle_format)
    # opponent = MaxBasePowerPlayer(battle_format=battle_format)

    # Start training
    await train_dqn(
        dqn_agent=player,
        opponent_player=opponent,
        num_episodes=training_params['num_episodes'],
        log_freq=training_params['log_freq'],
        save_freq=training_params['save_freq']
    )

if __name__ == "__main__":
    # Ensure you have a Pokémon Showdown server running locally on the configured port
    # (e.g., using `pokemon-showdown start --no-security`)
    print("Starting DQN Training...")
    # Check if event loop is already running (e.g., in Jupyter)
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is running, schedule main as a task
             loop.create_task(main())
             # If in Jupyter/IPython, this might just schedule it.
             # You might need await main() in an async cell.
        else:
            # If loop exists but not running, run until complete
            loop.run_until_complete(main())
    except RuntimeError:
        # If no event loop exists, create one and run until complete
        asyncio.run(main())

    print("Training script finished.")