import json
import os

def load_config(config_path="config.json"):
    """Load configuration from JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    return config

def get_server_config():
    """Get server configuration."""
    config = load_config()
    return config["server"]

def get_account_config(account_name):
    """Get account configuration."""
    config = load_config()
    if account_name not in config["accounts"]:
        raise ValueError(f"Account {account_name} not found in config")
    
    return config["accounts"][account_name]

def get_battle_format():
    """Get the battle format."""
    config = load_config()
    return config["battle_format"]