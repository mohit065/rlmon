# train2.py (Modified for RandomPlayer only)
import time
import asyncio
import logging
# import random # No longer needed for opponent selection

# Import the improved agent
from agent2 import DQNAgent
from poke_env.player import RandomPlayer # Only import RandomPlayer

# --- Configuration ---
BATTLE_FORMAT = "gen4randombattle" # Ensure this matches the agent's expectation if hardcoded elsewhere
NUM_EPISODES = 10000
LOG_FREQ = 10 # Log progress every 10 episodes
SAVE_FREQ = 500 # Save model every 500 episodes (adjust as needed)
MODEL_LOAD_PATH = None # Set to "dqn_model_v2.pth" or specific path to force loading, None uses agent default
MODEL_SAVE_PATH = None # Set to a specific path to save, None uses agent default (dqn_model_v2.pth)

# --- Logging Setup ---
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Training Function ---
# Modified to accept a single opponent instance
async def train_dqn(agent: DQNAgent, opponent: RandomPlayer, num_episodes: int):
    """Trains the DQNAgent against a single specified opponent."""
    start_time = time.time()
    total_steps_start = agent.steps_done # Track steps added during this training run
    opponent_name = opponent.__class__.__name__
    print(f"\nStarting training for {num_episodes} episodes against {opponent_name}...")
    print(f"Initial Steps Done: {agent.steps_done}")

    for episode in range(1, num_episodes + 1):
        # Opponent is fixed, no need to select from a pool
        # The agent's internal state (last_battle_state_vec, etc.) is reset
        # automatically within choose_move when a battle finishes.

        try:
            # print(f"\n--- Episode {episode}/{num_episodes} vs {opponent_name} ---")
            await agent.battle_against(opponent, n_battles=1)

            # Logging progress
            if episode % LOG_FREQ == 0:
                total_battles = agent.n_finished_battles
                current_wins = agent.n_won_battles # Agent tracks total wins across sessions if loaded
                win_rate = current_wins / total_battles if total_battles > 0 else 0.0
                elapsed_time = time.time() - start_time
                steps_this_run = agent.steps_done - total_steps_start
                print(f"Episode: {episode}/{num_episodes} | "
                      f"Steps (Total): {agent.steps_done} | "
                      f"Steps (Session): {steps_this_run} | "
                      f"Win Rate (Overall): {win_rate:.3f} ({current_wins}/{total_battles}) | "
                      f"Time: {elapsed_time:.2f}s")

            # Saving model periodically
            if episode % SAVE_FREQ == 0:
                agent.save_model(path=MODEL_SAVE_PATH) # Use agent's default path if None

        except Exception as e:
            # Log the full traceback for better debugging
            logger.error(f"Error during episode {episode} against {opponent_name}: {e}", exc_info=True)
            print(f"Error in episode {episode}, resetting battle state and continuing...")
            agent.reset_battles() # Reset battle count specific things if needed
            # Consider adding a small delay or specific error handling
            await asyncio.sleep(2) # Short delay after an error

    # Final save
    agent.save_model(path=MODEL_SAVE_PATH)
    end_time = time.time()
    total_steps_end = agent.steps_done
    print(f"\n--- Training Finished ---")
    print(f"Total Episodes: {num_episodes}")
    print(f"Total Steps Taken (Session): {total_steps_end - total_steps_start}")
    print(f"Total Steps (Overall): {total_steps_end}")
    final_win_rate = agent.n_won_battles / agent.n_finished_battles if agent.n_finished_battles > 0 else 0.0
    print(f"Final Overall Win Rate: {final_win_rate:.3f} ({agent.n_won_battles}/{agent.n_finished_battles})")
    print(f"Total Training Time: {end_time - start_time:.2f}s")

# --- Main Execution ---
async def main():
    print("Setting up Improved DQN agent (agent2)...")
    player = DQNAgent(battle_format=BATTLE_FORMAT)

    # Attempt to load a pre-existing model
    player.load_model(path=MODEL_LOAD_PATH) # Uses agent's default path if None

    # Define the single opponent
    opponent_random = RandomPlayer(battle_format=BATTLE_FORMAT)
    print(f"Training exclusively against: {opponent_random.__class__.__name__}")

    # Start the training loop, passing the single opponent instance
    await train_dqn(agent=player, opponent=opponent_random, num_episodes=NUM_EPISODES)

if __name__ == "__main__":
    print("Starting Improved DQN Training (train2.py - RandomPlayer Only)...")
    # Handle asyncio loop properly
    try:
        # Get the current event loop.
        loop = asyncio.get_event_loop()
        # Run the main coroutine until it completes.
        loop.run_until_complete(main())
    except RuntimeError as e:
        # If the above fails (e.g., "cannot run loop while another loop is running")
        # which can happen in certain environments like Jupyter notebooks,
        # try using asyncio.run() as a fallback.
        if "Cannot run the event loop while another loop is running" in str(e):
            print("RuntimeError detected, trying asyncio.run()...")
            asyncio.run(main())
        else:
            raise e # Re-raise other RuntimeErrors
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        # Attempt to save model one last time on exit, if player exists
        try:
            if 'player' in locals() and isinstance(player, DQNAgent):
                 print("\nAttempting final model save...")
                 player.save_model(path=MODEL_SAVE_PATH)
        except Exception as final_save_e:
            print(f"Could not perform final save: {final_save_e}")
        print("Training script finished.")