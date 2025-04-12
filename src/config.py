import os

SERVER_HOST = "localhost"
SERVER_PORT = 8000 # Default for local server, change if needed
AUTH_SERVER_URL = None # Use None for local server

ACCOUNTS = {
    "account1": {
        "username": "rlmon2",
        "password": "rlmon2"
    },
    "account2": {
        "username": "rlmonbot",
        "password": "rlmonbot"
    },
}

# Use a format with fixed team size if possible for simpler state/action spaces initially
# gen4randombattle is fine, but state/action spaces can vary wildly.
# Consider gen8randombattle for more modern features if desired later.
BATTLE_FORMAT = "gen4randombattle"

# DQN Hyperparameters (can be tuned)
STATE_DIM = 150       # Placeholder - adjust based on embed_battle!
ACTION_DIM = 10       # 4 moves + 6 switches (Max team size - 1 active)
HIDDEN_DIM = 128
LEARNING_RATE = 1e-4
GAMMA = 0.95         # Discount factor
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 10000 # Number of steps for epsilon to decay
TARGET_UPDATE_FREQ = 1000 # Steps between target network updates
REPLAY_BUFFER_SIZE = 50000
BATCH_SIZE = 64

# Training parameters
NUM_TRAINING_EPISODES = 50000
LOG_FREQ = 10 # Log progress every N episodes
SAVE_FREQ = 1000 # Save model every N episodes
MODEL_SAVE_PATH = "dqn_model.pth"


def load_config():
    return {
        "server": {
            "host": SERVER_HOST,
            "port": SERVER_PORT,
            "auth_server_url": AUTH_SERVER_URL
        },
        "accounts": ACCOUNTS,
        "battle_format": BATTLE_FORMAT,
        "dqn_params": {
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "hidden_dim": HIDDEN_DIM,
            "lr": LEARNING_RATE,
            "gamma": GAMMA,
            "epsilon_start": EPSILON_START,
            "epsilon_end": EPSILON_END,
            "epsilon_decay": EPSILON_DECAY,
            "target_update_freq": TARGET_UPDATE_FREQ,
            "buffer_size": REPLAY_BUFFER_SIZE,
            "batch_size": BATCH_SIZE
        },
        "training_params": {
            "num_episodes": NUM_TRAINING_EPISODES,
            "log_freq": LOG_FREQ,
            "save_freq": SAVE_FREQ,
            "model_save_path": MODEL_SAVE_PATH
        }
    }

def get_server_config():
    return {
        "host": SERVER_HOST,
        "port": SERVER_PORT,
        "auth_server_url": AUTH_SERVER_URL
    }

def get_account_config(account_name):
    if account_name not in ACCOUNTS:
        raise ValueError(f"Account {account_name} not found in config")
    return ACCOUNTS[account_name]

def get_battle_format():
    return BATTLE_FORMAT

def get_dqn_params():
    config = load_config()
    return config["dqn_params"]

def get_training_params():
    config = load_config()
    return config["training_params"]