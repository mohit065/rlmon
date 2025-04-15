import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import os
import logging

from poke_env.player import Player
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.environment.battle import Battle

from config import (
    get_account_config, get_server_config, get_battle_format,
    get_dqn_params, get_training_params
)

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

import random
from collections import deque, namedtuple

Experience = namedtuple('Experience',
                        ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, *args):
        """Saves an experience."""
        self.memory.append(Experience(*args))

    def sample(self, batch_size):
        """Randomly samples a batch of experiences from memory."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def is_ready(self, batch_size):
        return len(self.memory) >= batch_size

import logging

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Rest of your script ---
# (No need to set log_level in Player init or configure poke_env logger separately)


class DQNAgent(Player):
    def __init__(self, battle_format, account_name="account2", train_mode=True):
        account_config = get_account_config(account_name)
        server_config = get_server_config() # Using default localhost from config
        self.dqn_params = get_dqn_params()
        self.training_params = get_training_params()

        account_configuration = AccountConfiguration(
            username=account_config["username"],
            password=account_config["password"]
        )


        # If using LocalhostServerConfiguration, ensure host/port match config.py and your server setup
        # server_configuration = LocalhostServerConfiguration # Assuming default localhost:8000
        server_configuration = LocalhostServerConfiguration


        super().__init__(
            account_configuration=account_configuration,
            server_configuration=server_configuration, # Use ShowdownServerConfiguration even for local
            battle_format=battle_format,
        )

        self.state_dim = self.dqn_params['state_dim']
        self.action_dim = self.dqn_params['action_dim'] # 4 moves + 6 switches

        self.policy_net = DQN(self.state_dim, self.action_dim, self.dqn_params['hidden_dim']).to(device)
        self.target_net = DQN(self.state_dim, self.action_dim, self.dqn_params['hidden_dim']).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only for inference

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.dqn_params['lr'])
        self.memory = ReplayBuffer(self.dqn_params['buffer_size'])

        self.gamma = self.dqn_params['gamma']
        self.epsilon_start = self.dqn_params['epsilon_start']
        self.epsilon_end = self.dqn_params['epsilon_end']
        self.epsilon_decay = self.dqn_params['epsilon_decay']

        self.steps_done = 0
        self.train_mode = train_mode # Control exploration/exploitation/learning

        # For tracking state across turns
        self.last_battle_state = None
        self.last_action_idx = None

        # Load existing model if not in training mode or if continuing training
        model_path = self.training_params['model_save_path']
        if os.path.exists(model_path) and not train_mode:
             print(f"Loading model from {model_path}")
             self.policy_net.load_state_dict(torch.load(model_path, map_location=device))
             self.policy_net.eval() # Set to evaluation mode
             self.target_net.load_state_dict(self.policy_net.state_dict()) # Ensure target net is also updated
             self.target_net.eval()
             self.train_mode = False # Ensure no exploration if loading for testing
             print("Model loaded successfully for evaluation.")
        elif os.path.exists(model_path) and train_mode:
             print(f"Loading model from {model_path} to continue training...")
             self.policy_net.load_state_dict(torch.load(model_path, map_location=device))
             self.target_net.load_state_dict(self.policy_net.state_dict())
             # Might want to load optimizer state and steps_done too if saved
             print("Model loaded successfully for continued training.")

    # In dqn_agent.py, inside the DQNAgent class:

# ... (other methods like __init__, embed_battle, choose_move, etc.) ...

    def reward_computing_helper(
        self,
        battle: Battle,
        *,
        fainted_value: float = 0.15,
        hp_value: float = 0.15,
        number_of_pokemons: int = 6,
        starting_value: float = 0.0,
        status_value: float = 0.15,
        victory_value: float = 1.0,
        defeat_value: float = -1.0, # Added defeat value for completeness
    ) -> float:
        """
        Computes a reward value based on the battle state. Copied from poke-env Player class.

        Args:
            battle: The battle environment.
            fainted_value: The reward coefficient for fainting an opponent's Pokemon.
            hp_value: The reward coefficient for HP difference.
            number_of_pokemons: The number of Pokemons in the team.
            starting_value: The base reward value.
            status_value: The reward coefficient for status conditions.
            victory_value: The reward for winning the battle.
            defeat_value: The reward for losing the battle.


        Returns:
            The computed reward value.
        """
        if battle.won:
            return victory_value
        elif battle.lost:
            return defeat_value # Return negative reward for loss

        reward = starting_value

        # Calculate total HP functionalities
        current_hp = sum(
            pokemon.current_hp_fraction for pokemon in battle.team.values()
        )
        opponent_hp = sum(
            pokemon.current_hp_fraction for pokemon in battle.opponent_team.values()
        )

        # HP reward
        reward += (current_hp - opponent_hp) * hp_value

        # Fainted Pokemon reward
        current_fainted = sum(
            1 for pokemon in battle.team.values() if pokemon.fainted
        )
        opponent_fainted = sum(
            1 for pokemon in battle.opponent_team.values() if pokemon.fainted
        )
        reward += (opponent_fainted - current_fainted) * fainted_value

        # Status condition reward
        current_status = sum(
            1 for pokemon in battle.team.values() if pokemon.status is not None
        )
        opponent_status = sum(
            1 for pokemon in battle.opponent_team.values() if pokemon.status is not None
        )
        reward -= (current_status - opponent_status) * status_value

        # Ensure reward is clipped between defeat and victory values if needed
        reward = max(defeat_value, min(reward, victory_value))

        return reward



    def embed_battle(self, battle: Battle) -> np.ndarray:
        """
        Converts the complex Battle object into a fixed-size numpy array (state).
        This is a VERY simplified example. Needs significant improvement for real performance.
        Features included:
        - Active Pokemon: HP fraction, stats (normalized?), type(s) (one-hot) - Simplified here
        - Opponent's Active Pokemon: HP fraction (if known) - Simplified
        - Available Moves: Power (normalized), PP fraction - Simplified
        - Team Pokemon (Bench): HP fraction, status (simple flags) - Simplified
        """
        # --- Simplification ---
        # This embedding is rudimentary and likely insufficient.
        # It needs padding/masking for variable numbers of moves/pokemon/types etc.
        # Normalization is crucial.
        # Using a fixed size requires careful planning or complex padding/masking.

        state = np.zeros(self.state_dim)
        offset = 0

        # Active Pokemon features (Simplified: just HP fraction)
        if battle.active_pokemon:
            state[offset] = battle.active_pokemon.current_hp_fraction
        offset += 1
        # Add more features: stats, types, status, boosts... (e.g., 20 features)
        offset += 19 # Placeholder space

        # Opponent Active Pokemon features (Simplified: just HP fraction)
        if battle.opponent_active_pokemon:
            state[offset] = battle.opponent_active_pokemon.current_hp_fraction
        offset += 1
        # Add more features if known: stats, types, status... (e.g., 20 features)
        offset += 19 # Placeholder space

        # Available Moves features (Simplified: just PP fraction for 4 moves)
        for i in range(4):
            if i < len(battle.available_moves):
                move = battle.available_moves[i]
                state[offset] = move.current_pp / move.max_pp if move.max_pp > 0 else 0
                # Add more features: power, type, accuracy... (e.g., 10 features per move)
            offset += 1 # Add more offsets if adding features
        offset += (4 - len(battle.available_moves)) # Pad unused move slots
        offset += 4 * 9 # Placeholder space for more move features (10 total per move)

        # Team features (Simplified: HP fraction for 5 switchable Pokemon)
        switchable_mons = [p for p in battle.available_switches if p]
        for i in range(5): # Max 5 switches
            if i < len(switchable_mons):
                mon = switchable_mons[i]
                state[offset] = mon.current_hp_fraction
                # Add more features: status, types... (e.g., 5 features per mon)
            offset += 1 # Add more offsets if adding features
        offset += (5 - len(switchable_mons)) # Pad unused switch slots
        offset += 5 * 4 # Placeholder space for more bench features (5 total per mon)


        # --- Important ---
        # Ensure the total number of features matches self.state_dim
        # This example uses placeholders and is likely incorrect in size.
        # You MUST carefully design this function and adjust STATE_DIM in config.py
        # print(f"Final offset: {offset}, Expected state_dim: {self.state_dim}")
        if offset > self.state_dim:
            # Truncate if too long (bad practice, indicates wrong state_dim)
            state = state[:self.state_dim]
            # print(f"Warning: State vector truncated. Final offset {offset} > state_dim {self.state_dim}")
        elif offset < self.state_dim:
             # Pad if too short (better than truncating)
             pass # Already zero-padded

        # Normalize state features (example: divide by max values) if not done above
        # state = state / normalizers

        return state.astype(np.float32)

    def _action_to_move(self, action_idx: int, battle: Battle):
        """Converts an action index (0-9) to a PlayerOrder."""
        # Action space: 0-3 are moves, 4-9 are switches
        if 0 <= action_idx < 4:
            # It's a move
            if action_idx < len(battle.available_moves):
                return self.create_order(battle.available_moves[action_idx])
            else:
                # Chosen move index is invalid (e.g., less than 4 moves available)
                # Fallback: choose the first available move
                if battle.available_moves:
                    # print(f"Warning: Action index {action_idx} invalid for moves. Fallback to move 0.")
                    return self.create_order(battle.available_moves[0])
                else: # Or if trapped and can't move, maybe struggle or default? Poke-env might handle this.
                     # print(f"Warning: Action index {action_idx} chosen but no moves available. Defaulting.")
                     return self.choose_random_move(battle) # Fallback to random if truly stuck

        elif 4 <= action_idx < 10: # Indices 4 to 9 for switches
            # It's a switch
            switch_idx = action_idx - 4
            if switch_idx < len(battle.available_switches):
                 pokemon_to_switch = battle.available_switches[switch_idx]
                 # Ensure the selected pokemon is not fainted or already active (should be handled by available_switches)
                 if pokemon_to_switch and not pokemon_to_switch.active and not pokemon_to_switch.fainted:
                     return self.create_order(pokemon_to_switch)
                 else:
                    #Chosen switch is invalid (e.g. index out of bounds, points to fainted/active mon)
                    #Fallback: choose the first valid switch, or random move if no valid switch
                    valid_switches = [p for p in battle.available_switches if p and not p.fainted and not p.active]
                    if valid_switches:
                        # print(f"Warning: Action index {action_idx} invalid for switches. Fallback to first valid switch.")
                        return self.create_order(valid_switches[0])
                    elif battle.available_moves:
                         # print(f"Warning: Action index {action_idx} invalid for switches, no valid switches. Fallback to first move.")
                         return self.create_order(battle.available_moves[0])
                    else:
                         # print(f"Warning: Action index {action_idx} invalid, no valid switches or moves. Defaulting.")
                         return self.choose_random_move(battle) # Ultimate fallback

        else:
            # Invalid action index outside 0-9 range
            print(f"Error: Invalid action index {action_idx}. Choosing random move.")
            return self.choose_random_move(battle)


    def select_action(self, state_tensor: torch.Tensor, available_moves, available_switches) -> int:
        """Selects an action using epsilon-greedy policy."""
        sample = random.random()
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.steps_done / self.epsilon_decay)

        if self.train_mode:
            self.steps_done += 1

        if self.train_mode and sample < eps_threshold:
            # Exploration: Choose a random *valid* action index
            possible_actions = []
            if available_moves:
                possible_actions.extend(range(len(available_moves))) # Indices 0-3
            if available_switches:
                 valid_switch_indices = [i + 4 for i, p in enumerate(available_switches) if p and not p.fainted and not p.active]
                 possible_actions.extend(valid_switch_indices) # Indices 4-9

            if not possible_actions: # Should not happen if battle isn't over
                print("Warning: No valid actions found during exploration. Defaulting.")
                return 0 # Default to action 0 (first move or fallback)
            action_idx = random.choice(possible_actions)
            # print(f"Explore: Chose action {action_idx} (Epsilon: {eps_threshold:.2f})")
            return action_idx
        else:
            # Exploitation: Choose the best action from the policy network
            with torch.no_grad():
                # Get Q-values for the current state
                q_values = self.policy_net(state_tensor)

                # Filter Q-values to only consider valid actions
                valid_action_mask = torch.full_like(q_values, -float('inf'), device=device) # Mask invalid actions

                # Mark valid move actions
                for i in range(len(available_moves)):
                    if i < 4: # Ensure we don't exceed action dimension for moves
                         valid_action_mask[0, i] = q_values[0, i] # Use calculated Q-value for valid moves

                # Mark valid switch actions
                for i, p in enumerate(available_switches):
                     action_idx = i + 4
                     if action_idx < self.action_dim and p and not p.fainted and not p.active: # Ensure we don't exceed action dimension for switches
                         valid_action_mask[0, action_idx] = q_values[0, action_idx] # Use calculated Q-value

                # Check if any valid action exists based on the mask
                if torch.all(valid_action_mask == -float('inf')):
                    # This can happen if the network hasn't learned or if state is weird
                    # print("Warning: No valid actions found based on Q-values/mask. Choosing random valid action.")
                    possible_actions = []
                    if available_moves: possible_actions.extend(range(len(available_moves)))
                    if available_switches: possible_actions.extend([i + 4 for i, p in enumerate(available_switches) if p and not p.fainted and not p.active])
                    if not possible_actions: return 0 # Ultimate fallback
                    action_idx = random.choice(possible_actions)
                else:
                    # Select the action with the highest Q-value among valid actions
                    action_idx = valid_action_mask.argmax().item()

                # print(f"Exploit: Chose action {action_idx} (Q-values: {q_values.cpu().numpy()}, Masked: {valid_action_mask.cpu().numpy()})")
                return action_idx


    def compute_reward(self, battle: Battle) -> float:
        """
        Computes the reward based on the battle state.
        This is called *after* the turn has occurred.
        We compare the current state to the state before the opponent's move.
        """
        # Simple reward: +1 for winning, -1 for losing, 0 otherwise
        if battle.won:
            return 1.0
        elif battle.lost:
            return -1.0
        else:
             # Intermediate rewards (optional, can help training speed)
             # Example: Reward based on HP difference, fainted Pokemon count
             # This requires comparing to previous state, which is complex to get right.
             # Using poke-env's internal reward buffer (simpler):
             return self.reward_computing_helper(
                 battle, fainted_value=1.5, hp_value=0.15, victory_value=10.0
             )
             # return 0.0 # Simplest intermediate reward


    def learn(self):
        """Samples experiences from the buffer and updates the network."""
        if not self.memory.is_ready(self.dqn_params['batch_size']):
            return # Not enough samples yet

        experiences = self.memory.sample(self.dqn_params['batch_size'])
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043)
        batch = Experience(*zip(*experiences))

        # Convert batch arrays to tensors
        state_batch = torch.cat([torch.from_numpy(s).unsqueeze(0) for s in batch.state]).to(device)
        action_batch = torch.tensor(batch.action, device=device).unsqueeze(1) # LongTensor for gather
        reward_batch = torch.tensor(batch.reward, device=device, dtype=torch.float32)
        next_state_batch = torch.cat([torch.from_numpy(s).unsqueeze(0) for s in batch.next_state]).to(device)
        done_batch = torch.tensor(batch.done, device=device, dtype=torch.bool) # Boolean tensor


        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        # Q(s_t, a) values are computed using the policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non-final next states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the done mask: V(s) = 0 if s is a terminal state.
        next_state_values = torch.zeros(self.dqn_params['batch_size'], device=device)
        # Use target_net to compute Q values for the next states
        # We only need Q values for non-final states (where done_batch is False)
        non_final_mask = ~done_batch
        non_final_next_states = next_state_batch[non_final_mask]

        if non_final_next_states.size(0) > 0: # Check if there are any non-final states
             next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()


        # Compute the expected Q values (Bellman equation)
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss (or MSE loss)
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        # loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100) # Gradient clipping
        self.optimizer.step()

        # Update target network periodically
        if self.steps_done % self.dqn_params['target_update_freq'] == 0:
            # print("Updating target network...")
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item() # Return loss for logging


    def choose_move(self, battle: Battle):
        current_state_vec = self.embed_battle(battle)
        current_state_tensor = torch.from_numpy(current_state_vec).unsqueeze(0).to(device)

        # --- Store experience from the PREVIOUS turn ---
        # If this isn't the first turn of the battle (last_battle_state exists)
        # and we took an action (last_action_idx is not None)
        if self.last_battle_state is not None and self.last_action_idx is not None:
            reward = self.compute_reward(battle) # Reward is for the transition S -> S'
            done = battle.finished

            # Store (S_t, A_t, R_t+1, S_t+1, done)
            self.memory.push(self.last_battle_state, self.last_action_idx, reward, current_state_vec, done)

            # Trigger learning step if in training mode
            if self.train_mode:
                self.learn() # Learn based on stored experiences

        # --- Select action for the CURRENT turn ---
        action_idx = self.select_action(current_state_tensor, battle.available_moves, battle.available_switches)

        # --- Update state for the next turn ---
        # If the battle is over, reset the last state tracking
        if battle.finished:
            self.last_battle_state = None
            self.last_action_idx = None
            # print(f"Battle finished. Won: {battle.won}")
        else:
            # Store the current state and chosen action index for the next call
            self.last_battle_state = current_state_vec
            self.last_action_idx = action_idx

        # Convert the chosen action index back to a PlayerOrder
        order = self._action_to_move(action_idx, battle)
        # print(f"Turn {battle.turn}: Chose action index {action_idx} -> Order: {order}")
        return order


    def save_model(self, path=None):
        """Saves the policy network's state dictionary."""
        if path is None:
            path = self.training_params['model_save_path']
        print(f"Saving model to {path}...")
        torch.save(self.policy_net.state_dict(), path)
        print("Model saved.")

    # Override battle_finished to reset state tracking
    def _battle_finished_callback(self, battle):
        # print(f"Battle finished callback. ID: {battle.battle_tag}. Won: {battle.won}")
        self.last_battle_state = None
        self.last_action_idx = None
        # Potentially add end-of-episode logic here if needed