import random
import logging
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple

from poke_env.player import Player
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.environment.battle import Battle

STATE_DIM = 150
ACTION_DIM = 10
HIDDEN_DIM = 128
LEARNING_RATE = 1e-4
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 10000
TARGET_UPDATE_FREQ = 1000
REPLAY_BUFFER_SIZE = 50000
BATCH_SIZE = 64
USERNAME = "rlmonbot"
PASSWORD = "rlmonbot"
MODEL_PATH = "dqn_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
Experience = namedtuple('Experience',('state', 'action', 'reward', 'next_state', 'done'))
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, *args):
        self.memory.append(Experience(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def is_ready(self, batch_size):
        return len(self.memory) >= batch_size

class DQNAgent(Player):
    def __init__(self, battle_format):
        account_configuration = AccountConfiguration(username=USERNAME,password=PASSWORD)
        server_configuration = LocalhostServerConfiguration

        super().__init__(
            account_configuration=account_configuration,
            server_configuration=server_configuration,
            battle_format=battle_format,
        )
        self.policy_net = DQN(STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(device)
        self.target_net = DQN(STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayBuffer(REPLAY_BUFFER_SIZE)
        self.steps_done = 0
        self.last_battle_state = None
        self.last_action_idx = None

    def embed_battle(self, battle: Battle) -> np.ndarray:
        state = np.zeros(STATE_DIM)
        offset = 0
        if battle.active_pokemon:
            state[offset] = battle.active_pokemon.current_hp_fraction
        offset += 1
        offset += 19
        if battle.opponent_active_pokemon:
            state[offset] = battle.opponent_active_pokemon.current_hp_fraction
        offset += 1
        offset += 19
        for i in range(4):
            if i < len(battle.available_moves):
                move = battle.available_moves[i]
                state[offset] = move.current_pp / move.max_pp if move.max_pp > 0 else 0
            offset += 1
        offset += (4 - len(battle.available_moves))
        offset += 4 * 9
        switchable_mons = [p for p in battle.available_switches if p]
        for i in range(5):
            if i < len(switchable_mons):
                mon = switchable_mons[i]
                state[offset] = mon.current_hp_fraction
            offset += 1
        offset += (5 - len(switchable_mons))
        offset += 5 * 4
        if offset > STATE_DIM:
            state = state[:STATE_DIM]
        elif offset < STATE_DIM:
            pass
        return state.astype(np.float32)

    def _action_to_move(self, action_idx: int, battle: Battle):
        if 0 <= action_idx < 4:
            if action_idx < len(battle.available_moves):
                return self.create_order(battle.available_moves[action_idx])
            else:
                if battle.available_moves:
                    return self.create_order(battle.available_moves[0])
                else:
                    return self.choose_random_move(battle)

        elif 4 <= action_idx < 10:
            switch_idx = action_idx - 4
            if switch_idx < len(battle.available_switches):
                 pokemon_to_switch = battle.available_switches[switch_idx]
                 if pokemon_to_switch and not pokemon_to_switch.active and not pokemon_to_switch.fainted:
                    return self.create_order(pokemon_to_switch)
                 else:
                    valid_switches = [p for p in battle.available_switches if p and not p.fainted and not p.active]
                    if valid_switches:
                        return self.create_order(valid_switches[0])
                    elif battle.available_moves:
                        return self.create_order(battle.available_moves[0])
                    else:
                        return self.choose_random_move(battle)

        else:
            return self.choose_random_move(battle)

    def select_action(self, state_tensor: torch.Tensor, available_moves, available_switches) -> int:
        sample = random.random()
        eps_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-1. * self.steps_done / EPSILON_DECAY)
        self.steps_done += 1
        if sample < eps_threshold:
            possible_actions = []
            if available_moves:
                possible_actions.extend(range(len(available_moves)))
            if available_switches:
                valid_switch_indices = [i + 4 for i, p in enumerate(available_switches) if p and not p.fainted and not p.active]
                possible_actions.extend(valid_switch_indices)

            if not possible_actions:
                print("Error here: No possible actions") # this is an error
                return 0
            
            action_idx = random.choice(possible_actions)
            return action_idx
        
        else:
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                valid_action_mask = torch.full_like(q_values, -float('inf'), device=device)
                for i in range(len(available_moves)):
                    if i < 4:
                        valid_action_mask[0, i] = q_values[0, i]

                for i, p in enumerate(available_switches):
                     action_idx = i + 4
                     if action_idx < ACTION_DIM and p and not p.fainted and not p.active:
                        valid_action_mask[0, action_idx] = q_values[0, action_idx]

                if torch.all(valid_action_mask == -float('inf')):
                    possible_actions = []
                    if available_moves: possible_actions.extend(range(len(available_moves)))
                    if available_switches: possible_actions.extend([i + 4 for i, p in enumerate(available_switches) if p and not p.fainted and not p.active])
                    if not possible_actions: return 0
                    action_idx = random.choice(possible_actions)
                else:
                    action_idx = valid_action_mask.argmax().item()

                return action_idx

    def compute_reward(self, battle: Battle) -> float:
        if battle.won:
            return 10.0
        elif battle.lost:
            return -1.0

        fainted_value = 1.5
        hp_value = 0.15
        status_value = 0.15
        victory_value = 10.0
        defeat_value = -1.0
        starting_value = 0.0

        reward = starting_value
        current_hp = sum(pokemon.current_hp_fraction for pokemon in battle.team.values())
        opponent_hp = sum(pokemon.current_hp_fraction for pokemon in battle.opponent_team.values())
        reward += (current_hp - opponent_hp) * hp_value
        current_fainted = sum(1 for pokemon in battle.team.values() if pokemon.fainted)
        opponent_fainted = sum(1 for pokemon in battle.opponent_team.values() if pokemon.fainted)
        reward += (opponent_fainted - current_fainted) * fainted_value
        current_status = sum(1 for pokemon in battle.team.values() if pokemon.status is not None)
        opponent_status = sum(1 for pokemon in battle.opponent_team.values() if pokemon.status is not None)
        reward -= (current_status - opponent_status) * status_value
        reward = max(defeat_value, min(reward, victory_value))
        return reward

    def learn(self):
        if not self.memory.is_ready(BATCH_SIZE):
            return

        experiences = self.memory.sample(BATCH_SIZE)
        batch = Experience(*zip(*experiences))
        state_batch = torch.cat([torch.from_numpy(s).unsqueeze(0) for s in batch.state]).to(device)
        action_batch = torch.tensor(batch.action, device=device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, device=device, dtype=torch.float32)
        next_state_batch = torch.cat([torch.from_numpy(s).unsqueeze(0) for s in batch.next_state]).to(device)
        done_batch = torch.tensor(batch.done, device=device, dtype=torch.bool)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        non_final_mask = ~done_batch
        non_final_next_states = next_state_batch[non_final_mask]
        if non_final_next_states.size(0) > 0:
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.steps_done % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def choose_move(self, battle: Battle):
        current_state_vec = self.embed_battle(battle)
        current_state_tensor = torch.from_numpy(current_state_vec).unsqueeze(0).to(device)
        if self.last_battle_state is not None and self.last_action_idx is not None:
            reward = self.compute_reward(battle)
            done = battle.finished
            self.memory.push(self.last_battle_state, self.last_action_idx, reward, current_state_vec, done)
            self.learn()

        action_idx = self.select_action(current_state_tensor, battle.available_moves, battle.available_switches)
        if battle.finished:
            self.last_battle_state = None
            self.last_action_idx = None
        else:
            self.last_battle_state = current_state_vec
            self.last_action_idx = action_idx

        order = self._action_to_move(action_idx, battle)
        return order

    def save_model(self, path=None):
        if path is None:
            path = MODEL_PATH
        print(f"Saving model to {path}...")
        torch.save(self.policy_net.state_dict(), path)
        print("Model saved.")