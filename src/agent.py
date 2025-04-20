import random
import numpy as np
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from poke_env.player import Player
from poke_env.environment.battle import Battle
from poke_env.environment.status import Status as S
from poke_env.environment.pokemon_type import PokemonType as T

STATE_DIM = 277             # Dimension of the state embedding
ACTION_DIM = 9              # Dimension of the action space
HIDDEN_DIM = 256            # Dimension of the hidden layers in the DQN
LEARNING_RATE = 1e-4        # Learning rate for the Adam optimizer
GAMMA = 0.95                # Discount factor for future rewards
EPSILON_START = 1.0         # Starting value for exploration rate
EPSILON_END = 0.05          # Minimum value for exploration rate
EPSILON_DECAY = 20000       # Decay steps for exploration rate
TARGET_UPDATE_FREQ = 1000   # How often to update the target network (in steps)
REPLAY_BUFFER_SIZE = 50000  # Capacity of the replay buffer
BATCH_SIZE = 64             # Batch size for sampling from the replay buffer
MAX_GRAD_NORM = 1.0         # Maximum norm for gradient clipping

WIN_REWARD = 10.0           # Reward obtained on winning
LOSS_REWARD = -10.0         # Reward obtained on losing
FAINT_WEIGHT = 1.5          # Reward factor for fainted pokemon
HP_WEIGHT = 0.2             # Reward factor for HP damage
STATUS_WEIGHT = 0.1         # Reward factor for status conditions

TYPE_MAP = { T.NORMAL: 0, T.FIRE: 1, T.WATER: 2, T.ELECTRIC: 3, T.GRASS: 4, T.ICE: 5, T.FIGHTING: 6, 
    T.POISON: 7, T.GROUND: 8, T.FLYING: 9,T.PSYCHIC: 10, T.BUG: 11, T.ROCK: 12, T.GHOST: 13, 
    T.DRAGON: 14, T.DARK: 15, T.STEEL: 16,}

STATUS_MAP = {S.BRN: 1, S.PAR: 2, S.PSN: 3, S.SLP: 4, S.FRZ: 5,}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Experience = namedtuple('Experience',('state', 'action', 'reward', 'next_state', 'done'))

class DQN(nn.Module):
    """Deep Q-Network model"""

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
    """Fixed-size buffer to store experience tuples"""

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
    """DQN Agent for Pokemon battles"""

    def __init__(self, battle_format, team):
        super().__init__(battle_format=battle_format, team=team)

        self.policy_net = DQN(STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(device)
        self.target_net = DQN(STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayBuffer(REPLAY_BUFFER_SIZE)
        self.steps_done = 0
        self.last_battle_state = None
        self.last_action_idx = None

    def embed_battle(self, battle: Battle):
        """Creates a state vector representation of the current battle state"""

        state = np.full(STATE_DIM, -1.0, dtype=np.float32)
        offset = 0

        # Player Team Information
        player_team = list(battle.team.values())
        player_active_idx = -1.0
        player_team_ordered = [None] * 6
        for i, p in enumerate(player_team):
            if i < 6: player_team_ordered[i] = p

        # Player HP Fractions (6 elements)
        for i in range(6):
            pokemon = player_team_ordered[i]
            state[offset] = pokemon.current_hp_fraction if pokemon else 0.0
            offset += 1

        # Player Types
        for i in range(6):
            pokemon = player_team_ordered[i]
            if pokemon and pokemon.types:
                type1, type2 = pokemon.types
                state[offset] = TYPE_MAP.get(type1, -1.0)
                offset += 1
                state[offset] = TYPE_MAP.get(type2, -1.0) if type2 else state[offset-1]
                offset += 1
            else:
                state[offset:offset+2] = -1.0
                offset += 2

        # Player Move Details
        for i in range(6):
            pokemon = player_team_ordered[i]
            if pokemon:
                moves = list(pokemon.moves.values())
                for j in range(4):
                    if j < len(moves):
                        move = moves[j]
                        state[offset] = TYPE_MAP.get(move.type, -1.0)
                        offset += 1
                        state[offset] = move.base_power / 250.0 if move.base_power > 0 else 0.0
                        offset += 1
                        state[offset] = move.accuracy
                        offset += 1
                        state[offset] = move.current_pp/move.max_pp if move.max_pp > 0 else 0.0
                        offset += 1
                    else:
                        state[offset:offset+4] = -1.0
                        offset += 4
            else:
                state[offset:offset+16] = -1.0
                offset += 16

        # Player Status Conditions
        for i in range(6):
            pokemon = player_team_ordered[i]
            if pokemon:
                state[offset] = STATUS_MAP.get(pokemon.status, 0.0) if pokemon.status else 0.0
            offset += 1

        # Player Current Stats
        stats_order = ['atk', 'def', 'spa', 'spd', 'spe']
        for i in range(6):
            pokemon = player_team_ordered[i]
            if pokemon and pokemon.stats:
                for stat_name in stats_order:
                    stat_value = pokemon.stats.get(stat_name, 0)
                    if isinstance(stat_value, (int, float)):
                        state[offset] = stat_value / 714.0
                    else:
                        state[offset] = -1.0
                    offset += 1
                if pokemon.active:
                    player_active_idx = float(i)
            else:
                state[offset:offset+5] = -1.0
                offset += 5

        # Player Active Pokémon Index
        state[offset] = player_active_idx
        offset += 1

        # Opponent's Team Information (Known Parts)
        opponent_team = list(battle.opponent_team.values())
        opponent_active_idx = -1.0
        opponent_team_ordered = [None] * 6

        for i, p in enumerate(opponent_team):
            if i < 6: opponent_team_ordered[i] = p

        # Opponent HP Fractions (6 elements)
        for i in range(6):
            pokemon = opponent_team_ordered[i]
            if pokemon:
                state[offset] = pokemon.current_hp_fraction
            offset += 1

        # Opponent Types
        for i in range(6):
            pokemon = opponent_team_ordered[i]
            if pokemon and pokemon.types:
                type1, type2 = pokemon.types
                state[offset] = TYPE_MAP.get(type1, -1.0)
                offset += 1
                state[offset] = TYPE_MAP.get(type2, -1.0) if type2 else state[offset-1]
                offset += 1
            else:
                state[offset:offset+2] = -1.0
                offset += 2

        # Opponent Move Details
        for i in range(6):
            pokemon = opponent_team_ordered[i]
            if pokemon:
                known_moves = list(pokemon.moves.values())
                for j in range(4):
                    if j < len(known_moves):
                        move = known_moves[j]
                        state[offset] = TYPE_MAP.get(move.type, -1.0)
                        offset += 1
                        state[offset] = move.base_power / 250.0 if move.base_power > 0 else 0.0
                        offset += 1
                        state[offset] = move.accuracy
                        offset += 1
                        state[offset] = -1.0
                        offset += 1
                    else:
                        state[offset:offset+4] = -1.0
                        offset += 4
            else:
                state[offset:offset+16] = -1.0
                offset += 16

        # Opponent Status Conditions
        for i in range(6):
            pokemon = opponent_team_ordered[i]
            if pokemon:
                state[offset] = STATUS_MAP.get(pokemon.status, 0.0) if pokemon.status else 0.0
            offset += 1

        # Opponent Active Boosts
        opponent_active_pokemon = battle.opponent_active_pokemon
        if opponent_active_pokemon:
            for idx, p in enumerate(opponent_team_ordered):
                if p and p == opponent_active_pokemon:
                    opponent_active_idx = float(idx)
                    break
            boosts = opponent_active_pokemon.boosts
            for stat_name in stats_order:
                state[offset] = boosts.get(stat_name, 0) / 6.0
                offset += 1
        else:
            state[offset:offset+5] = -1.0
            offset += 5

        # Opponent Active Pokémon Index
        state[offset] = opponent_active_idx
        offset += 1

        # Final Check
        if offset != STATE_DIM:
            if offset < STATE_DIM:
                state[offset:] = -1.0
            elif offset > STATE_DIM:
                state = state[:STATE_DIM]

        return state

    def _action_to_move(self, action_idx: int, battle: Battle):
        """Converts an action index (0-8) into a Pokemon object order"""

        # Choose a move
        if 0 <= action_idx < 4:
            if action_idx < len(battle.available_moves):
                return self.create_order(battle.available_moves[action_idx])
            else:
                if battle.available_moves:
                    return self.create_order(battle.available_moves[0])
                else:
                    return self.choose_random_move(battle)

        # Choose a pokemon
        elif 4 <= action_idx < ACTION_DIM:
            target_switch_idx = action_idx - 4
            valid_switch_options = []
            for i, p in enumerate(battle.available_switches):
                if i < 5 and p and not p.fainted and not p.active:
                    valid_switch_options.append((i, p))

            chosen_pokemon = None
            for original_idx, pokemon in valid_switch_options:
                if original_idx == target_switch_idx:
                    chosen_pokemon = pokemon
                    break

            if chosen_pokemon:
                return self.create_order(chosen_pokemon)
            elif valid_switch_options:
                return self.create_order(valid_switch_options[0][1])
            elif battle.available_moves:
                return self.create_order(battle.available_moves[0])
            else:
                return self.choose_random_move(battle)
            
        else:
            return self.choose_random_move(battle)

    def select_action(self, state_tensor: torch.Tensor, available_moves, available_switches):
        """Selects an action using an epsilon-greedy policy."""

        possible_actions = []
        if available_moves:
            possible_actions.extend(range(min(len(available_moves), 4)))

        if available_switches:
            for i, p in enumerate(available_switches):
                action_idx = i + 4
                if action_idx < ACTION_DIM and p and not p.fainted and not p.active:
                    possible_actions.append(action_idx)

        if not possible_actions:
            return random.choice(possible_actions)

        # Epsilon-greedy approach
        sample = random.random()
        eps_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-1. * self.steps_done / EPSILON_DECAY)
        self.steps_done += 1

        # Exploration
        if sample < eps_threshold:
            action_idx = random.choice(possible_actions)

        # Exploitation
        else:
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                valid_action_mask = torch.full_like(q_values, -float('inf'), device=device)
                for idx in possible_actions:
                    valid_action_mask[0, idx] = q_values[0, idx]

                if torch.all(valid_action_mask == -float('inf')):
                   action_idx = random.choice(possible_actions)
                else:
                   action_idx = valid_action_mask.argmax().item()

        return action_idx

    def compute_reward(self, battle: Battle):
        """Computes the reward based on the current battle state."""

        # Primary reward
        if battle.won:
            return WIN_REWARD
        if battle.lost:
            return LOSS_REWARD

        reward = 0.0
        opponent_team = battle.opponent_team.values()
        opponent_fainted_count = 0
        opponent_hp_sum_fraction = 0.0
        opponent_status_count = 0
        num_opponent_pokemon = len(opponent_team) if len(opponent_team) > 0 else 6

        # Find fainted or status-afflicted opponent pokemon
        for p in opponent_team:
            if p.fainted:
                opponent_fainted_count += 1
            opponent_hp_sum_fraction += p.current_hp_fraction
            if p.status is not None:
                opponent_status_count += 1

        # Secondary rewards based on damage, knockouts and status conditions
        reward += opponent_fainted_count * FAINT_WEIGHT
        opponent_missing_hp_fraction = max(0.0, num_opponent_pokemon - opponent_hp_sum_fraction)
        reward += opponent_missing_hp_fraction * HP_WEIGHT
        reward += opponent_status_count * STATUS_WEIGHT
        player_fainted_count = sum(1 for p in battle.team.values() if p.fainted)
        reward -= player_fainted_count * FAINT_WEIGHT

        return reward

    def learn(self):
        """
        Samples experiences from the replay buffer, computes the DDQN loss,
        performs backpropagation with gradient clipping, updates the policy network,
        and periodically updates the target network.
        """

        # Skip learning if not enough samples in replay buffer
        if not self.memory.is_ready(BATCH_SIZE):
            return None

        # Sample a batch of experiences and unpack into separate tensors
        experiences = self.memory.sample(BATCH_SIZE)
        batch = Experience(*zip(*experiences))

        # Convert states and next states to tensors
        state_batch = torch.cat([torch.from_numpy(s).unsqueeze(0) for s in batch.state]).to(device)
        next_state_batch = torch.cat([torch.from_numpy(s).unsqueeze(0) for s in batch.next_state]).to(device)

        # Convert actions, rewards, and dones to tensors
        action_batch = torch.tensor(batch.action, dtype=torch.long, device=device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=device)
        done_batch = torch.tensor(batch.done, dtype=torch.bool, device=device)

        # Compute Q(s, a) from the policy network
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Double DQN: Use policy net to find best next actions, target net to evaluate them
        with torch.no_grad():
            policy_next_q_vals = self.policy_net(next_state_batch)
            best_next_actions = policy_next_q_vals.argmax(1).unsqueeze(1)

            target_next_q_vals = self.target_net(next_state_batch)
            next_state_q_values_target = target_next_q_vals.gather(1, best_next_actions).squeeze(1)

            # Zero-out the Q-values where next state is terminal
            next_state_q_values_target[done_batch] = 0.0

            # Compute expected Q values using Bellman equation
            expected_state_action_values = (next_state_q_values_target * GAMMA) + reward_batch

        # Compute Huber (smooth L1) loss between predicted and target Q-values
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Backpropagation with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=MAX_GRAD_NORM)
        self.optimizer.step()

        # Update target network periodically
        if self.steps_done % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


    def choose_move(self, battle: Battle):
        """
        Embeds state, calculates reward for previous transition, stores experience,
        performs learning step, selects action, and returns the chosen order.
        """

        # Convert current battle state into a vector and tensor
        current_state_vec = self.embed_battle(battle)
        current_state_tensor = torch.from_numpy(current_state_vec).unsqueeze(0).to(device)

        # Compute reward from the current battle state
        reward = self.compute_reward(battle)

        # If there was a previous state and action, store the transition and learn from it
        if self.last_battle_state is not None and self.last_action_idx is not None:
            done = battle.finished
            self.memory.push(self.last_battle_state, self.last_action_idx, reward, current_state_vec, done)
            self.learn()

        # Choose an action index using the current policy
        action_idx = self.select_action(current_state_tensor, battle.available_moves, battle.available_switches)

        # If the battle is over, clear the stored state and action
        if battle.finished:
            self.last_battle_state = None
            self.last_action_idx = None
        else:
            # Otherwise, update the last state and action for the next step
            self.last_battle_state = current_state_vec
            self.last_action_idx = action_idx

        # Convert the chosen action index into a battle order and return it
        order = self._action_to_move(action_idx, battle)
        return order
    
    def save_model(self, path):
        """Saves the policy and target networks along with optimizer state and step count."""

        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done
        }
        torch.save(checkpoint, path)

    def load_model(self, path):
        """Loads the policy and target networks, optimizer state, and step count."""

        checkpoint = torch.load(path, map_location=device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint.get('steps_done', 0)