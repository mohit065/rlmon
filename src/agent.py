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

# --- Hyperparameters and Constants ---
STATE_DIM = 277             # Dimension of the state embedding
ACTION_DIM = 9              # Dimension of the action space (4 moves + 5 switches)
HIDDEN_DIM = 256            # Dimension of the hidden layers in the DQN
LEARNING_RATE = 1e-4        # Learning rate for the Adam optimizer
GAMMA = 0.95                # Discount factor for future rewards
EPSILON_START = 1.0         # Starting value for epsilon (exploration rate)
EPSILON_END = 0.05          # Minimum value for epsilon
EPSILON_DECAY = 20000       # Decay rate for epsilon (higher means slower decay)
TARGET_UPDATE_FREQ = 1000   # How often to update the target network (in steps)
REPLAY_BUFFER_SIZE = 50000  # Capacity of the replay buffer
BATCH_SIZE = 64             # Batch size for sampling from the replay buffer
MAX_GRAD_NORM = 1.0         # Maximum norm for gradient clipping
MODEL_PATH = "model.pth"

# Reward Weights (tune these)
WIN_REWARD = 10.0
LOSS_REWARD = -10.0
FAINT_WEIGHT = 1.5       # +/- reward per fainted pokemon difference
HP_WEIGHT = 0.2          # Reward factor for opponent's missing HP
STATUS_WEIGHT = 0.1      # Reward factor per opponent status condition

# Normalization Constants for embed_battle
MAX_STAT = 714.0         # Approximate max possible base stat * modifier (adjust if needed)
MAX_POWER = 250.0        # Max move base power (adjust for specific gen/moves)
MAX_BOOST = 6.0          # Max stat stage boost

# --- Mappings ---
# Type mapping (Gen 1-5 standard types)
TYPE_MAP = { T.NORMAL: 0, T.FIRE: 1, T.WATER: 2, T.ELECTRIC: 3, T.GRASS: 4, T.ICE: 5, T.FIGHTING: 6, 
    T.POISON: 7, T.GROUND: 8, T.FLYING: 9,T.PSYCHIC: 10, T.BUG: 11, T.ROCK: 12, T.GHOST: 13, 
    T.DRAGON: 14, T.DARK: 15, T.STEEL: 16,}

# Status mapping (map None to 0 later)
STATUS_MAP = {S.BRN: 1, S.PAR: 2, S.PSN: 3, S.SLP: 4, S.FRZ: 5,}

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
Experience = namedtuple('Experience',('state', 'action', 'reward', 'next_state', 'done'))

# --- DQN Network Definition ---
class DQN(nn.Module):
    """Deep Q-Network model."""
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Output raw Q-values (no activation)
        return self.fc3(x)

# --- Replay Buffer ---
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, *args):
        """Save an experience."""
        self.memory.append(Experience(*args))

    def sample(self, batch_size):
        """Sample a batch of experiences."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def is_ready(self, batch_size):
        """Check if buffer contains enough samples for a batch."""
        return len(self.memory) >= batch_size

# --- DQN Agent ---
class DQNAgent(Player):
    """DQN Agent for Pokemon battles."""
    def __init__(self, battle_format, team):
        super().__init__(battle_format=battle_format, team=team)

        # Initialize Networks
        self.policy_net = DQN(STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(device)
        self.target_net = DQN(STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target network is only for inference

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)

        # Replay Memory
        self.memory = ReplayBuffer(REPLAY_BUFFER_SIZE)

        # Learning step counter
        self.steps_done = 0

        # Tracking the last state and action for storing transitions
        self.last_battle_state = None
        self.last_action_idx = None

    def embed_battle(self, battle: Battle) -> np.ndarray:
        """
        Creates a state vector representation of the current battle state.

        Args:
            battle: The current Battle object from poke_env.

        Returns:
            A numpy array representing the battle state, padded/truncated to STATE_DIM.
            Uses -1.0 as a placeholder for unknown or non-applicable values.
        """
        # Initialize state vector with placeholder value
        state = np.full(STATE_DIM, -1.0, dtype=np.float32)
        offset = 0

        # --- Player's Team Information ---
        player_team = list(battle.team.values())
        player_active_idx = -1.0 # Placeholder for active Pokémon index (0-5)
        player_team_ordered = [None] * 6 # Ensure 6 slots representation
        for i, p in enumerate(player_team):
            if i < 6: player_team_ordered[i] = p

        # 1. Player HP Fractions (6 elements)
        for i in range(6):
            pokemon = player_team_ordered[i]
            state[offset] = pokemon.current_hp_fraction if pokemon else 0.0
            offset += 1

        # 2. Player Types (6 pokémon * 2 types = 12 elements)
        for i in range(6):
            pokemon = player_team_ordered[i]
            if pokemon and pokemon.types:
                type1, type2 = pokemon.types
                state[offset] = TYPE_MAP.get(type1, -1.0) # Use global TYPE_MAP
                offset += 1
                state[offset] = TYPE_MAP.get(type2, -1.0) if type2 else state[offset-1]
                offset += 1
            else:
                state[offset:offset+2] = -1.0 # Fill both slots
                offset += 2

        # 3. Player Move Details (6 pokémon * 4 moves * 4 features = 96 elements)
        for i in range(6):
            pokemon = player_team_ordered[i]
            if pokemon:
                moves = list(pokemon.moves.values())
                for j in range(4): # Iterate through 4 move slots
                    if j < len(moves):
                        move = moves[j]
                        state[offset] = TYPE_MAP.get(move.type, -1.0)
                        offset += 1
                        state[offset] = move.base_power / MAX_POWER if move.base_power > 0 else 0.0
                        offset += 1
                        state[offset] = move.accuracy # poke-env accuracy is 0-1
                        offset += 1
                        state[offset] = move.current_pp/move.max_pp if move.max_pp > 0 else 0.0 # PP Fraction
                        offset += 1
                    else: # Empty move slot
                        state[offset:offset+4] = -1.0
                        offset += 4
            else: # No pokemon in this slot
                state[offset:offset+16] = -1.0
                offset += 16

        # 4. Player Status Conditions (6 elements)
        for i in range(6):
            pokemon = player_team_ordered[i]
            if pokemon:
                # Use global STATUS_MAP, map None to 0.0
                state[offset] = STATUS_MAP.get(pokemon.status, 0.0) if pokemon.status else 0.0
            # Keep -1.0 placeholder if no pokemon in this slot
            offset += 1

        # 5. Player Current Stats (6 pokémon * 5 stats = 30 elements)
        stats_order = ['atk', 'def', 'spa', 'spd', 'spe']
        for i in range(6):
            pokemon = player_team_ordered[i]
            if pokemon and pokemon.stats:
                for stat_name in stats_order:
                    # Normalize stats
                    # Get the stat value, defaulting to 0 if missing
                    stat_value = pokemon.stats.get(stat_name, 0)
                    # Check if the value is actually numeric before dividing
                    if isinstance(stat_value, (int, float)):
                        state[offset] = stat_value / MAX_STAT
                    else:
                        state[offset] = -1.0
                    offset += 1
                if pokemon.active:
                    player_active_idx = float(i) # Record the index (0-5) if active
            else: # No pokemon or stats known
                state[offset:offset+5] = -1.0
                offset += 5

        # 6. Player Active Pokémon Index (1 element)
        state[offset] = player_active_idx
        offset += 1

        # --- Opponent's Team Information (Known Parts) ---
        opponent_team = list(battle.opponent_team.values())
        opponent_active_idx = -1.0 # Placeholder for opponent active index
        opponent_team_ordered = [None] * 6 # Ensure 6 slots representation

        for i, p in enumerate(opponent_team):
            if i < 6: opponent_team_ordered[i] = p

        # 7. Opponent HP Fractions (6 elements)
        for i in range(6):
            pokemon = opponent_team_ordered[i]
            if pokemon: # Only fill if we've seen the Pokemon
                state[offset] = pokemon.current_hp_fraction # Will be 0.0 if fainted
            # Keep -1.0 placeholder if never seen
            offset += 1

        # 8. Opponent Types (6 pokémon * 2 types = 12 elements)
        for i in range(6):
            pokemon = opponent_team_ordered[i]
            if pokemon and pokemon.types: # Only fill if seen and types known
                type1, type2 = pokemon.types
                state[offset] = TYPE_MAP.get(type1, -1.0)
                offset += 1
                state[offset] = TYPE_MAP.get(type2, -1.0) if type2 else state[offset-1]
                offset += 1
            else: # Unseen or unknown types
                 state[offset:offset+2] = -1.0
                 offset += 2

        # 9. Opponent Move Details (6 pokémon * 4 moves * 4 features = 96 elements)
        for i in range(6):
            pokemon = opponent_team_ordered[i]
            if pokemon: # Only process known pokemon
                known_moves = list(pokemon.moves.values()) # Moves revealed so far
                for j in range(4): # Iterate through 4 potential move slots
                    if j < len(known_moves):
                         move = known_moves[j]
                         state[offset] = TYPE_MAP.get(move.type, -1.0)
                         offset += 1
                         state[offset] = move.base_power / MAX_POWER if move.base_power > 0 else 0.0
                         offset += 1
                         state[offset] = move.accuracy
                         offset += 1
                         state[offset] = -1.0 # Opponent PP Fraction is unknown
                         offset += 1
                    else: # Unknown move slot
                         state[offset:offset+4] = -1.0
                         offset += 4
            else: # Unseen pokemon slot
                 state[offset:offset+16] = -1.0
                 offset += 16

        # 10. Opponent Status Conditions (6 elements)
        for i in range(6):
            pokemon = opponent_team_ordered[i]
            if pokemon: # Only fill if pokemon is known
                state[offset] = STATUS_MAP.get(pokemon.status, 0.0) if pokemon.status else 0.0
            # Keep -1.0 placeholder if pokemon never seen
            offset += 1

        # 11. Opponent Active Boosts (5 elements)
        opponent_active_pokemon = battle.opponent_active_pokemon
        if opponent_active_pokemon:
            # Try to find index of opponent active mon in our ordered list
            for idx, p in enumerate(opponent_team_ordered):
                if p and p == opponent_active_pokemon: # Precise object comparison
                    opponent_active_idx = float(idx)
                    break
            # Record boosts regardless of finding index
            boosts = opponent_active_pokemon.boosts
            for stat_name in stats_order:
                # Normalize boosts to [-1, 1] range
                state[offset] = boosts.get(stat_name, 0) / MAX_BOOST
                offset += 1
        else: # No opponent active pokemon
            state[offset:offset+5] = -1.0
            offset += 5

        # 12. Opponent Active Pokémon Index (1 element)
        state[offset] = opponent_active_idx
        offset += 1

        # --- Final Check & Return ---
        if offset != STATE_DIM:
            print(f"Embed Battle: Final offset ({offset}) != STATE_DIM ({STATE_DIM}). Check logic.")
            # Pad or truncate if necessary, though ideally offset should match
            if offset < STATE_DIM:
                state[offset:] = -1.0 # Pad with placeholders
            elif offset > STATE_DIM:
                 state = state[:STATE_DIM] # Truncate

        return state

    def _action_to_move(self, action_idx: int, battle: Battle):
        """
        Converts an action index (0-8) into a poke_env Move or Pokemon object order.
        Includes validation and fallbacks for robustness.

        Args:
            action_idx: The integer action index chosen by the agent.
            battle: The current Battle object.

        Returns:
            A poke_env Order object (Move or Pokemon).
        """
        # Action indices 0-3 correspond to available moves
        if 0 <= action_idx < 4:
            if action_idx < len(battle.available_moves):
                # Valid move index chosen
                return self.create_order(battle.available_moves[action_idx])
            else:
                # Agent chose a move index not currently available (e.g., only 2 moves possible)
                # Fallback: Choose the first available move if possible
                if battle.available_moves:
                    print(f"Action {action_idx} invalid for moves {battle.available_moves}. Fallback to move 0.")
                    return self.create_order(battle.available_moves[0])
                else:
                    # No moves available at all (e.g., must Struggle)
                    print(f"Action {action_idx} chosen but no moves available. Fallback to random.")
                    return self.choose_random_move(battle)

        # Action indices 4-8 correspond to available switches (pokemon 0-4 on bench)
        elif 4 <= action_idx < ACTION_DIM:
            target_switch_idx = action_idx - 4 # Maps action 4-8 to switch list index 0-4

            # Find all genuinely valid switch options among the first 6 slots
            valid_switch_options = []
            for i, p in enumerate(battle.available_switches):
                # Ensure we only consider indices relevant to our action space (0-4 bench mons)
                # and that the Pokemon is valid for switching
                if i < 5 and p and not p.fainted and not p.active:
                    valid_switch_options.append((i, p)) # Store (original_list_index, pokemon_object)

            chosen_pokemon = None
            # Check if the agent's target index corresponds to one of the valid options found
            for original_idx, pokemon in valid_switch_options:
                if original_idx == target_switch_idx:
                    chosen_pokemon = pokemon
                    break

            if chosen_pokemon:
                # Successfully found the specific valid switch targeted by the agent
                return self.create_order(chosen_pokemon)
            elif valid_switch_options:
                # Agent's target index wasn't valid/available, but other switches are.
                # Fallback: Pick the first valid switch found.
                print(f"Action {action_idx} invalid for switches {battle.available_switches}. Fallback to valid switch {valid_switch_options[0][0]}.")
                return self.create_order(valid_switch_options[0][1])
            elif battle.available_moves:
                 # No valid switches at all, try available moves
                 print(f"Action {action_idx} chosen but no valid switches. Fallback to move 0.")
                 return self.create_order(battle.available_moves[0])
            else:
                # No valid switches or moves, must use random (likely Struggle)
                print(f"Action {action_idx} chosen but no valid switches or moves. Fallback to random.")
                return self.choose_random_move(battle)

        # Action index is out of expected range (shouldn't happen if select_action is correct)
        else:
            print(f"Invalid action index {action_idx} received. Falling back to random move.")
            return self.choose_random_move(battle)

    def select_action(self, state_tensor: torch.Tensor, available_moves, available_switches) -> int:
        """
        Selects an action using an epsilon-greedy policy.
        Filters actions based on validity in the current battle state.

        Args:
            state_tensor: The current state represented as a PyTorch tensor.
            available_moves: List of currently available moves.
            available_switches: List of currently available switches.

        Returns:
            The integer index of the chosen action (0-8).
        """
        # --- 1. Determine all currently valid action indices (0-8) ---
        possible_actions = []
        # Moves (indices 0-3)
        if available_moves:
            possible_actions.extend(range(min(len(available_moves), 4))) # Cap at index 3

        # Switches (indices 4-8, corresponding to bench slots 0-4)
        if available_switches:
            for i, p in enumerate(available_switches):
                action_idx = i + 4
                # Check if switch corresponds to action space (4-8) and is valid
                if action_idx < ACTION_DIM and p and not p.fainted and not p.active:
                    possible_actions.append(action_idx)

        # --- 2. Handle edge case: No actions possible ---
        if not possible_actions:
            return random.choice(possible_actions)

        # --- 3. Epsilon-greedy decision ---
        sample = random.random()
        # Calculate epsilon based on global constants
        eps_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                        np.exp(-1. * self.steps_done / EPSILON_DECAY)
        self.steps_done += 1 # Increment step counter *after* using it for epsilon

        if sample < eps_threshold:
            # --- 4. Exploration: Choose randomly from valid actions ---
            action_idx = random.choice(possible_actions)
            # print(f"Exploring: Chose action {action_idx} from {possible_actions}")
        else:
            # --- 5. Exploitation: Choose best action based on Q-values ---
            with torch.no_grad(): # Inference mode
                q_values = self.policy_net(state_tensor)
                # Create mask: Initialize all actions to -infinity
                valid_action_mask = torch.full_like(q_values, -float('inf'), device=device)
                # Set Q-values for valid actions in the mask
                for idx in possible_actions:
                    valid_action_mask[0, idx] = q_values[0, idx]

                # Check if all valid actions were masked to -inf (unlikely but possible)
                if torch.all(valid_action_mask == -float('inf')):
                   # Fallback: Choose randomly if network gives no preference among valid actions
                   print("Exploiting, but all valid Q-values were -inf. Choosing randomly.")
                   action_idx = random.choice(possible_actions)
                else:
                   # Choose the action with the highest Q-value among valid actions
                   action_idx = valid_action_mask.argmax().item()
                   # print(f"Exploiting: Chose action {action_idx} (Q: {valid_action_mask[0, action_idx]:.2f}) from {possible_actions}")

        return action_idx

    def compute_reward(self, battle: Battle) -> float:
        """
        Computes the reward based on the current battle state.
        Prioritizes winning/losing, then considers fainted counts,
        opponent missing HP, and opponent status conditions.

        Args:
            battle: The current Battle object.

        Returns:
            The calculated reward value (float).
        """
        # --- 1. Terminal Rewards (Highest Priority) ---
        if battle.won:
            return WIN_REWARD
        if battle.lost:
            return LOSS_REWARD

        # --- 2. Intermediate Rewards (Calculated if game hasn't ended) ---
        reward = 0.0

        # Calculate opponent team stats
        opponent_team = battle.opponent_team.values()
        opponent_fainted_count = 0
        opponent_hp_sum_fraction = 0.0
        opponent_status_count = 0
        # Assume 6 opponent Pokemon if team size isn't fully known yet
        num_opponent_pokemon = len(opponent_team) if len(opponent_team) > 0 else 6

        for p in opponent_team:
            if p.fainted:
                opponent_fainted_count += 1
            # Sum known HP fractions
            opponent_hp_sum_fraction += p.current_hp_fraction
            if p.status is not None:
                opponent_status_count += 1

        # a) Reward for fainted opponents
        reward += opponent_fainted_count * FAINT_WEIGHT

        # b) Reward for opponent's missing HP (difference from max possible HP)
        opponent_missing_hp_fraction = max(0.0, num_opponent_pokemon - opponent_hp_sum_fraction)
        reward += opponent_missing_hp_fraction * HP_WEIGHT

        # c) Reward for opponent status conditions
        reward += opponent_status_count * STATUS_WEIGHT

        # d) Penalty for player's fainted Pokemon (encourage survival)
        player_fainted_count = sum(1 for p in battle.team.values() if p.fainted)
        # Use the same weight for symmetry, making it a penalty
        reward -= player_fainted_count * FAINT_WEIGHT

        return reward

    def choose_move(self, battle: Battle):
        """
        Main method called by poke_env to choose an action.
        Embeds state, calculates reward for previous transition, stores experience,
        performs learning step, selects action, and returns the chosen order.

        Args:
            battle: The current Battle object.

        Returns:
            A poke_env Order object.
        """
        # 1. Embed the current battle state
        current_state_vec = self.embed_battle(battle)
        current_state_tensor = torch.from_numpy(current_state_vec).unsqueeze(0).to(device)

        # 2. Calculate reward based on the outcome of the *previous* action
        # This reward is associated with the state we are transitioning *from*
        reward = self.compute_reward(battle)

        # 3. Store the transition (S, A, R, S') in the replay buffer
        # Only possible if this isn't the very first turn
        if self.last_battle_state is not None and self.last_action_idx is not None:
            done = battle.finished # Is the *current* state terminal?
            self.memory.push(self.last_battle_state, self.last_action_idx, reward, current_state_vec, done)
            _ = self.learn()

        # 5. Select action for the *current* state
        action_idx = self.select_action(current_state_tensor, battle.available_moves, battle.available_switches)

        # 6. Update tracker for the *next* transition
        if battle.finished:
             # Reset if the battle ended after this turn
             self.last_battle_state = None
             self.last_action_idx = None
        else:
            # Store current state and chosen action for the next iteration
            self.last_battle_state = current_state_vec
            self.last_action_idx = action_idx

        # 7. Convert action index to poke_env order
        order = self._action_to_move(action_idx, battle)
        return order

    def learn(self):
        """
        Samples experiences from the replay buffer, computes the DDQN loss,
        performs backpropagation with gradient clipping, updates the policy network,
        and periodically updates the target network.

        Returns:
            The scalar loss value for the batch, or None if learning did not occur.
        """
        # Ensure buffer has enough samples for a batch
        if not self.memory.is_ready(BATCH_SIZE):
            return None # Not enough memory yet

        # --- 1. Sample experiences ---
        experiences = self.memory.sample(BATCH_SIZE)
        batch = Experience(*zip(*experiences)) # Transpose batch

        # --- 2. Convert batch data to tensors ---
        state_batch = torch.cat([torch.from_numpy(s).unsqueeze(0) for s in batch.state]).to(device)
        action_batch = torch.tensor(batch.action, dtype=torch.long, device=device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=device)
        next_state_batch = torch.cat([torch.from_numpy(s).unsqueeze(0) for s in batch.next_state]).to(device)
        done_batch = torch.tensor(batch.done, dtype=torch.bool, device=device)

        # --- 3. Calculate Q(s, a) using the policy network ---
        # Q-values for the actions actually taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # --- 4. Calculate Double DQN Target values: V(s') = r + gamma * Q_target(s', argmax_a' Q_policy(s', a')) ---
        with torch.no_grad(): # Disable gradient calculation for target computation
            # a) Select best action in next state using the *policy* network
            policy_next_q_vals = self.policy_net(next_state_batch)
            best_next_actions = policy_next_q_vals.argmax(1).unsqueeze(1) # Get indices of max Q-values

            # b) Evaluate these selected actions using the *target* network
            target_next_q_vals = self.target_net(next_state_batch)
            # Get the Q-value from target_net corresponding to the action chosen by policy_net
            next_state_q_values_target = target_next_q_vals.gather(1, best_next_actions).squeeze(1) # Remove action dim

            # c) Zero out Q-values for terminal states (next state value is 0 if done)
            next_state_q_values_target[done_batch] = 0.0

            # d) Compute the Bellman target value
            expected_state_action_values = (next_state_q_values_target * GAMMA) + reward_batch

        # --- 5. Calculate Loss (Smooth L1 Loss / Huber Loss) ---
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # --- 6. Optimize the model ---
        self.optimizer.zero_grad()  # Clear old gradients
        loss.backward()             # Calculate gradients

        # --- 7. Gradient Clipping ---
        # Prevent exploding gradients by clipping the norm
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=MAX_GRAD_NORM)

        self.optimizer.step()       # Update policy network weights

        # --- 8. Update Target Network ---
        # Periodically copy weights from policy_net to target_net
        if self.steps_done % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item() # Return loss value for logging

    def save_model(self, path=MODEL_PATH):
        """Saves the policy network's state dictionary."""
        try:
            torch.save(self.policy_net.state_dict(), path)
        except Exception as e:
            print(f"Error saving model to {path}: {e}")

    def load_model(self, path=MODEL_PATH):
        """Loads the policy network's state dictionary."""
        try:
            self.policy_net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            # Sync target network after loading
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.policy_net.eval() # Set to evaluation mode if loading for inference
            self.target_net.eval()
        except FileNotFoundError:
            print(f"Model file not found at {path}. Starting with random weights.")
        except Exception as e:
            print(f"Error loading model from {path}: {e}")