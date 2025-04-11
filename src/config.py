SERVER_HOST = "localhost"
SERVER_PORT = 8000
AUTH_SERVER_URL = None

ACCOUNTS = {
    "account1": {
        "username": "rlmon2",
        "password": "rlmon2"
    },
    "account2": {
        "username": "rlmonbot",
        "password": "rlmonbot"
    }
}

BATTLE_FORMAT = "gen4randombattle"

def load_config():
    return {
        "server": {
            "host": SERVER_HOST,
            "port": SERVER_PORT,
            "auth_server_url": AUTH_SERVER_URL
        },
        "accounts": ACCOUNTS,
        "battle_format": BATTLE_FORMAT
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
