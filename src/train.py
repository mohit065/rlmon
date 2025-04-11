import numpy as np
import tensorflow as tf
import asyncio
import matplotlib.pyplot as plt
from poke_env.player import RandomPlayer
from poke_env.player_configuration import PlayerConfiguration
from poke_env.server_configuration import ServerConfiguration
from model import DQNAgent
from spaces import PokemonEnvironment
from config import get_account_config, get_server_config

# Hyperparameters
NUM_EPISODES = 1000
EVAL_EVERY = 100
SAVE_EVERY = 200
BATTLE_FORMAT = "gen8randombattle"
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995
GAMMA = 0.99
LEARNING_RATE = 0.0005
MEMORY_SIZE = 50000
BATCH_SIZE = 64
MODEL_PATH = "models/pokemon_dqn.h5"

# This function will be used to evaluate the agent
async def evaluate_agent(agent, opponent, n_battles=100):
    # Reset battle statistics
    agent.reset_battles()
    opponent.reset_battles()
    
    # Play n_battles games
    await agent.battle_against(opponent, n_battles=n_battles)
    
    # Calculate win rate
    win_rate = agent.n_won_battles / n_battles
    
    return win_rate

async def train_agent():
    print("Setting up training environment...")
    
    # Load account configs
    agent_config = get_account_config("account1")
    opponent_config = get_account_config("account2")
    server_config = get_server_config()
    
    # Set up configurations
    agent_player_config = PlayerConfiguration(
        username=agent_config["username"],
        password=agent_config["password"]
    )
    
    opponent_player_config = PlayerConfiguration(
        username=opponent_config["username"],
        password=opponent_config["password"]
    )
    
    server_configuration = ServerConfiguration(
        **server_config
    )

    # Create our DQN agent
    agent = DQNAgent(
        battle_format=BATTLE_FORMAT,
        player_configuration=agent_player_config,
        server_configuration=server_configuration,
        epsilon=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
        gamma=GAMMA,
        learning_rate=LEARNING_RATE,
        memory_size=MEMORY_SIZE,
        batch_size=BATCH_SIZE,
        model_path=MODEL_PATH
    )
    
    # Create the random opponent
    opponent = RandomPlayer(
        battle_format=BATTLE_FORMAT,
        player_configuration=opponent_player_config,
        server_configuration=server_configuration
    )
    
    # Create a second opponent for evaluation
    eval_opponent = RandomPlayer(
        battle_format=BATTLE_FORMAT,
        player_configuration=opponent_player_config,
        server_configuration=server_configuration
    )
    
    # Create the environment
    env = PokemonEnvironment(opponent, battle_format=BATTLE_FORMAT)
    
    print("Starting training...")
    
    # Initialize tracking variables
    rewards = []
    win_rates = []
    episodes = []
    
    # Main training loop
    for episode in range(NUM_EPISODES):
        # Reset battle for new episode
        battle_id = f"battle_{episode}"
        
        # Start a new battle
        battle = await agent.create_battle(
            battle_format=BATTLE_FORMAT,
            opponent=opponent.username,
            battle_id=battle_id
        )
        
        done = False
        total_reward = 0
        
        # Initialize current state
        current_state = env.embed_battle(battle)
        
        # Battle loop
        while not done:
            # Choose action
            action = agent.choose_move(battle)
            
            # Determine action index
            action_idx = 0
            if action in battle.available_moves:
                action_idx = battle.available_moves.index(action)
            elif action in battle.available_switches:
                action_idx = 4 + battle.available_switches.index(action)
            
            # Take action
            await agent.move(action)
            
            # Wait for opponent's move
            await asyncio.sleep(0.1)
            
            # Get new state and reward
            next_state = env.embed_battle(battle)
            reward = env.compute_reward(battle)
            done = battle.ended
            
            # Store transition in memory
            agent.remember(current_state, action_idx, reward, next_state, done)
            
            # Update current state
            current_state = next_state
            
            # Update total reward
            total_reward += reward
            
            # Train the agent
            agent.replay()
            
        # Track reward
        rewards.append(total_reward)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{NUM_EPISODES} completed. Epsilon: {agent.epsilon:.4f}")
        
        # Evaluate agent
        if (episode + 1) % EVAL_EVERY == 0:
            win_rate = await evaluate_agent(agent, eval_opponent, n_battles=20)
            print(f"Episode {episode + 1}: Win rate against random player: {win_rate:.4f}")
            episodes.append(episode + 1)
            win_rates.append(win_rate)
        
        # Save model
        if (episode + 1) % SAVE_EVERY == 0:
            agent.save_model()
    
    # Final evaluation
    win_rate = await evaluate_agent(agent, eval_opponent, n_battles=100)
    print(f"Final win rate against random player: {win_rate:.4f}")
    
    # Save final model
    agent.save_model()
    
    # Plot training progress
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(episodes, win_rates)
    plt.title('Win Rate vs Random Player')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.show()
    
    print("Training completed!")

if __name__ == "__main__":
    # Run the training
    asyncio.get_event_loop().run_until_complete(train_agent())