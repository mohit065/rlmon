# agent2.py
import random
import logging
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple
import copy # Needed for snapshotting battle state parts

from poke_env.player import Player
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.environment.battle import Battle
from poke_env.environment.pokemon_type import PokemonType
from poke_env.environment.move_category import MoveCategory
from poke_env.data import GenData # Using GenData for type chart access

# --- Constants ---
# Increased state dimension - adjust if features change significantly
# Estimate: (ActiveMon + OppActiveMon + 4*Moves + 5*Switches + Field)
# ~ (60 + 60 + 4*40 + 5*50 + 20) = ~550. Let's use 600 for buffer.
NEW_STATE_DIM = 600
ACTION_DIM = 10 # 4 moves + 6 switches (indices 0-3 are moves, 4-9 are switches)
HIDDEN_DIM = 256 # Increased hidden layer size for larger state
LEARNING_RATE = 1e-4 # Keep initial LR, tune later
GAMMA = 0.99 # Higher gamma for potentially longer battles
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 50000 # Increased decay steps, tune later
TARGET_UPDATE_FREQ = 1000
REPLAY_BUFFER_SIZE = 100000 # Increased buffer size
BATCH_SIZE = 64
USERNAME = "rlmonbot"
PASSWORD = "rlmonbot"
MODEL_PATH = "dqn_model_v2.pth" # New model path

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
Experience = namedtuple('Experience',('state', 'action', 'reward', 'next_state', 'done'))
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Type Information ---
# Add a flag to track if we are using the enum or fallback strings
USING_ENUM_TYPES = False
try:
    # Get type list from poke_env's PokemonType enum
    # Check if UNKNOWN exists before trying to use it in list comprehension
    has_unknown = hasattr(PokemonType, 'UNKNOWN')
    if has_unknown:
        TYPE_LIST = [t.name for t in PokemonType if t != PokemonType.UNKNOWN]
        TYPE_MAPPING = {t.name: i for i, t in enumerate(PokemonType) if t != PokemonType.UNKNOWN}
    else: # If UNKNOWN doesn't exist, just get all types
        print("Warning: PokemonType.UNKNOWN not found in enum. Including all types.")
        TYPE_LIST = [t.name for t in PokemonType]
        TYPE_MAPPING = {t.name: i for i, t in enumerate(PokemonType)}

    N_TYPES = len(TYPE_LIST)
    print(f"Successfully loaded {N_TYPES} types from poke_env.PokemonType: {TYPE_LIST}")
    USING_ENUM_TYPES = True # Set flag if successful
except Exception as e:
    print(f"Error getting types from poke_env.PokemonType ({e}), using fallback list.")
    # Fallback for older poke-env or different structure
    TYPE_LIST = ['NORMAL', 'FIRE', 'WATER', 'ELECTRIC', 'GRASS', 'ICE', 'FIGHTING', 'POISON', 'GROUND', 'FLYING', 'PSYCHIC', 'BUG', 'ROCK', 'GHOST', 'DRAGON', 'DARK', 'STEEL', 'FAIRY'] # Gen 6+
    TYPE_MAPPING = {t: i for i, t in enumerate(TYPE_LIST)}
    N_TYPES = len(TYPE_LIST)
    USING_ENUM_TYPES = False # Ensure flag is False on fallback

STATUS_LIST = ["BRN", "FRZ", "PAR", "PSN", "SLP", "TOX"] # Major statuses
N_STATUS = len(STATUS_LIST)

WEATHER_LIST = ["SunnyDay", "RainDance", "Sandstorm", "Hail"] # Add others if needed (e.g., DesolateLand)
N_WEATHER = len(WEATHER_LIST)


# --- Neural Network ---
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

# --- Replay Buffer (Unchanged) ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, *args):
        self.memory.append(Experience(*args))

    def sample(self, batch_size):
        # Handle case where buffer isn't full enough for the batch size
        actual_batch_size = min(batch_size, len(self.memory))
        return random.sample(self.memory, actual_batch_size)

    def __len__(self):
        return len(self.memory)

    def is_ready(self, batch_size):
        return len(self.memory) >= batch_size

# --- DQN Agent ---
class DQNAgent(Player):
    def __init__(self, battle_format):
        account_configuration = AccountConfiguration(username=USERNAME,password=PASSWORD)
        server_configuration = LocalhostServerConfiguration

        super().__init__(
            account_configuration=account_configuration,
            server_configuration=server_configuration,
            battle_format=battle_format,
        )
        self.policy_net = DQN(NEW_STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(device)
        self.target_net = DQN(NEW_STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayBuffer(REPLAY_BUFFER_SIZE)
        self.steps_done = 0

        # State for reward calculation
        self.last_battle_state_vec = None # Store the vector representation
        self.last_action_idx = None
        self.last_potential = 0.0
        self.previous_battle_snapshot = None # Store dict with relevant counts/values

        # Reward weights (tune these)
        self.hp_potential_weight = 0.05
        self.fainted_potential_weight = 0.3
        self.status_potential_weight = 0.0 # Status potential can be noisy, start low
        self.ko_reward = 0.5
        self.faint_penalty = 0.5
        self.win_reward = 10.0
        self.lose_penalty = -10.0 # Make losing more impactful


    # --- Helper Functions for Embedding ---
    def _one_hot_encode(self, value, item_list, mapping):
        vector = np.zeros(len(item_list))
        if value in mapping:
            vector[mapping[value]] = 1.0
        return vector

    # CORRECTED VERSION of _encode_types
    def _encode_types(self, pokemon):
        type1_vec = np.zeros(N_TYPES)
        type2_vec = np.zeros(N_TYPES) # Keep two vectors for consistent state size
        if pokemon and pokemon.types: # Check the .types tuple
            # --- Encode Type 1 ---
            type1_obj = pokemon.types[0]
            if type1_obj: # Check the type object exists
                type1_name = type1_obj.name
                if type1_name in TYPE_MAPPING:
                    type1_vec[TYPE_MAPPING[type1_name]] = 1.0

            # --- Encode Type 2 (if it exists) ---
            if len(pokemon.types) > 1:
                type2_obj = pokemon.types[1]
                if type2_obj: # Check the type object exists
                    type2_name = type2_obj.name
                    if type2_name in TYPE_MAPPING:
                        # Put the second type encoding in the second vector part
                        type2_vec[TYPE_MAPPING[type2_name]] = 1.0

        return np.concatenate([type1_vec, type2_vec])

    def _normalize_boosts(self, boosts):
        # Normalize boosts from -6 to +6 -> approx [0, 1]
        norm_boosts = np.zeros(6) # Atk, Def, SpA, SpD, Spe, Acc/Eva? (poke-env tracks 5 main)
        stat_order = ['atk', 'def', 'spa', 'spd', 'spe']
        for i, stat in enumerate(stat_order):
            boost_val = boosts.get(stat, 0)
            norm_boosts[i] = (boost_val + 6) / 12.0
        # Add placeholder for 6th value if needed, or adjust size to 5
        return norm_boosts

    # --- State Embedding ---
    def embed_battle(self, battle: Battle) -> np.ndarray:
        state = np.zeros(NEW_STATE_DIM)
        offset = 0

        # --- Own Active Pokemon ---
        mon = battle.active_pokemon
        if mon:
            state[offset] = mon.current_hp_fraction
            offset += 1
            state[offset:offset + 2 * N_TYPES] = self._encode_types(mon)
            offset += 2 * N_TYPES
            state[offset:offset + 6] = self._normalize_boosts(mon.boosts) # 5 stats + 1 placeholder? Check poke-env boost keys
            offset += 6
            status_vec = self._one_hot_encode(mon.status.name if mon.status else None, STATUS_LIST, {s: i for i, s in enumerate(STATUS_LIST)})
            state[offset:offset + N_STATUS] = status_vec
            offset += N_STATUS
            state[offset] = 1.0 if mon.is_dynamaxed else 0.0 # Add Dynamax if relevant for format
            offset += 1
            # Add other volatile statuses if needed (Confusion, etc.) - requires ~10-20 more features
        else: # Pad if no active mon (shouldn't happen mid-battle)
            offset += (1 + 2 * N_TYPES + 6 + N_STATUS + 1)

        # --- Opponent Active Pokemon ---
        opp_mon = battle.opponent_active_pokemon
        if opp_mon:
            state[offset] = opp_mon.current_hp_fraction
            offset += 1
            state[offset:offset + 2 * N_TYPES] = self._encode_types(opp_mon)
            offset += 2 * N_TYPES
            # We often don't know opponent boosts unless revealed
            # state[offset:offset + 6] = self._normalize_boosts(opp_mon.boosts) # Could add if known
            offset += 6 # Pad for boosts
            status_vec = self._one_hot_encode(opp_mon.status.name if opp_mon.status else None, STATUS_LIST, {s: i for i, s in enumerate(STATUS_LIST)})
            state[offset:offset + N_STATUS] = status_vec
            offset += N_STATUS
            state[offset] = 1.0 if opp_mon.is_dynamaxed else 0.0 # Add Dynamax if relevant
            offset += 1
        else: # Pad if no opponent active mon
             offset += (1 + 2 * N_TYPES + 6 + N_STATUS + 1)

        # --- Available Moves (Max 4) ---
        moves = battle.available_moves
        for i in range(4):
            if i < len(moves):
                move = moves[i]
                state[offset] = move.current_pp / move.max_pp if move.max_pp > 0 else 0
                offset += 1
                state[offset] = (move.base_power / 150.0) if move.base_power > 0 else 0 # Normalize power
                offset += 1
                state[offset] = (move.accuracy / 100.0) if isinstance(move.accuracy, (int, float)) else 1.0 # Accuracy (handle True)
                offset += 1
                state[offset:offset + N_TYPES] = self._one_hot_encode(move.type.name if move.type else None, TYPE_LIST, TYPE_MAPPING)
                offset += N_TYPES
                state[offset] = 1.0 if move.category == MoveCategory.PHYSICAL else 0.0
                offset += 1
                state[offset] = 1.0 if move.category == MoveCategory.SPECIAL else 0.0
                offset += 1
                state[offset] = 1.0 if move.category == MoveCategory.STATUS else 0.0
                offset += 1

                # Calculate effectiveness against opponent's active mon
                eff_multiplier = 1.0 # Default: neutral effectiveness

                # Check if opponent exists, has types, move exists, and move has a known type
                # Modify the type check based on whether we are using the enum or fallback strings
                is_known_type = False
                if move and move.type:
                    if USING_ENUM_TYPES:
                        # Check against the enum member if enums loaded correctly
                        # Need to ensure PokemonType.UNKNOWN exists if USING_ENUM_TYPES is True
                        if hasattr(PokemonType, 'UNKNOWN'):
                             is_known_type = (move.type != PokemonType.UNKNOWN)
                        else: # If UNKNOWN doesn't exist in enum, assume any type found is known
                             is_known_type = True
                    else:
                        # Check against the string name if using fallback
                        # Ensure move.type.name exists before comparing
                        if hasattr(move.type, 'name'):
                            is_known_type = (move.type.name != 'UNKNOWN') # Assuming 'UNKNOWN' is the name used
                        else:
                             is_known_type = False # Cannot determine type name


                if opp_mon and opp_mon.types and is_known_type: # Use the is_known_type flag here
                    try:
                        # USE THIS METHOD: Get multiplier from the opponent Pokemon perspective
                        eff_multiplier = opp_mon.damage_multiplier(move)
                    except Exception as e:
                        # Log error if damage_multiplier fails for some reason
                        # print(f"Warning: Could not calculate damage multiplier for move {move.id} against {opp_mon.species}. Error: {e}")
                        eff_multiplier = 1.0 # Default to neutral on error

                # Normalize the multiplier (range 0 to 4) to roughly [0, 1]
                normalized_eff = eff_multiplier / 4.0
                state[offset] = normalized_eff
                offset += 1
                # Add other move flags if needed (priority, contact, etc.) - adds ~5-10 features per move
            else:
                # Pad for non-existent moves
                offset += (1 + 1 + 1 + N_TYPES + 1 + 1 + 1 + 1) # Match features per move

        # --- Available Switches (Max 5 others) ---
        switches = battle.available_switches
        team_mons = list(battle.team.values()) # Get all team members
        switchable_mons = [p for p in team_mons if p in switches and not p.active] # Filter for actual switchable options

        for i in range(5): # Assuming max 5 benched mons (6 total)
            if i < len(switchable_mons):
                mon = switchable_mons[i]
                state[offset] = mon.current_hp_fraction
                offset += 1
                state[offset:offset + 2 * N_TYPES] = self._encode_types(mon)
                offset += 2 * N_TYPES
                status_vec = self._one_hot_encode(mon.status.name if mon.status else None, STATUS_LIST, {s: i for i, s in enumerate(STATUS_LIST)})
                state[offset:offset + N_STATUS] = status_vec
                offset += N_STATUS
                state[offset] = 1.0 if mon.fainted else 0.0 # Explicit fainted flag
                offset += 1
            else:
                # Pad for non-existent switches
                offset += (1 + 2 * N_TYPES + N_STATUS + 1) # Match features per switch

        # --- Battlefield State ---
        # Weather
        weather_vec = np.zeros(N_WEATHER)
        if battle.weather:
            # Ensure weather is not empty before accessing first element
            weather_keys = list(battle.weather.keys())
            if weather_keys:
                weather_name = weather_keys[0].name # Get name of active weather
                if weather_name in WEATHER_LIST:
                     weather_vec[WEATHER_LIST.index(weather_name)] = 1.0
        state[offset:offset + N_WEATHER] = weather_vec
        offset += N_WEATHER

        # Hazards (Own Side)
        state[offset] = 1.0 if "stealthrock" in battle.side_conditions else 0.0
        offset += 1
        state[offset] = battle.side_conditions.get("spikes", 0) / 3.0 # Normalize spike layers
        offset += 1
        state[offset] = battle.side_conditions.get("toxicspikes", 0) / 2.0 # Normalize tspike layers
        offset += 1
        state[offset] = 1.0 if "stickyweb" in battle.side_conditions else 0.0 # Add sticky web if gen6+
        offset += 1

        # Hazards (Opponent Side)
        state[offset] = 1.0 if "stealthrock" in battle.opponent_side_conditions else 0.0
        offset += 1
        state[offset] = battle.opponent_side_conditions.get("spikes", 0) / 3.0
        offset += 1
        state[offset] = battle.opponent_side_conditions.get("toxicspikes", 0) / 2.0
        offset += 1
        state[offset] = 1.0 if "stickyweb" in battle.opponent_side_conditions else 0.0
        offset += 1

        # Screens (Own Side)
        state[offset] = 1.0 if "lightscreen" in battle.side_conditions else 0.0
        offset += 1
        state[offset] = 1.0 if "reflect" in battle.side_conditions else 0.0
        offset += 1
        state[offset] = 1.0 if "auroraveil" in battle.side_conditions else 0.0 # Gen 7+
        offset += 1

        # Screens (Opponent Side)
        state[offset] = 1.0 if "lightscreen" in battle.opponent_side_conditions else 0.0
        offset += 1
        state[offset] = 1.0 if "reflect" in battle.opponent_side_conditions else 0.0
        offset += 1
        state[offset] = 1.0 if "auroraveil" in battle.opponent_side_conditions else 0.0
        offset += 1

        # Other Field Effects (Trick Room, Tailwind etc.) - Add if needed, ~5-10 features

        # Final check and padding/truncating
        if offset > NEW_STATE_DIM:
            # This should not happen if NEW_STATE_DIM is calculated correctly
            print(f"Warning: Actual state dimension ({offset}) exceeded NEW_STATE_DIM ({NEW_STATE_DIM}). Truncating.")
            state = state[:NEW_STATE_DIM]
        elif offset < NEW_STATE_DIM:
            # Pad with zeros if offset is smaller (e.g., if some features were skipped)
            state[offset:] = 0.0
            # print(f"Debug: Final offset {offset}, padding {NEW_STATE_DIM - offset} zeros.")


        # Ensure the final state has the correct dimension
        if len(state) != NEW_STATE_DIM:
             # If truncated, it will match. If padded, it should match.
             # If it still doesn't match, there's a logic error.
             raise ValueError(f"Final state dimension {len(state)} does not match NEW_STATE_DIM {NEW_STATE_DIM}")


        return state.astype(np.float32)

    # --- Action Conversion (Small Refinement) ---
    def _action_to_move(self, action_idx: int, battle: Battle):
        # Moves are indices 0-3
        if 0 <= action_idx < 4:
            if action_idx < len(battle.available_moves):
                move = battle.available_moves[action_idx]
                # Basic check: Don't select a move with 0 PP if others are available
                # unless the selected move *is* struggle
                if move.current_pp == 0 and move.id != 'struggle' and any(m.current_pp > 0 for m in battle.available_moves):
                     # Find the first valid move with PP > 0
                     valid_moves = [m for m in battle.available_moves if m.current_pp > 0]
                     if valid_moves:
                         print(f"Warning: Action {action_idx} selected move {move.id} with 0 PP. Switching to {valid_moves[0].id}")
                         return self.create_order(valid_moves[0])
                     else: # If all moves have 0 PP (e.g., Imprisoned) but struggle wasn't chosen
                         print(f"Warning: Action {action_idx} selected move {move.id} with 0 PP, but no other moves have PP. Forcing Struggle via default.")
                         return self.choose_default_move() # Pass battle object
                else:
                    return self.create_order(move)
            else:
                # Fallback if action index is out of bounds for available moves
                # This can happen if the selected action was invalid (e.g., from default choice)
                print(f"Warning: Action {action_idx} is invalid move index (max {len(battle.available_moves)-1}). Choosing default move.")
                return self.choose_default_move() # Pass battle object

        # Switches are indices 4-9 (representing team slots 1-6, excluding active)
        elif 4 <= action_idx < 10:
            switch_target_idx = action_idx - 4 # Corresponds to indices 0-5 in the team list
            team_list = list(battle.team.values())

            if switch_target_idx < len(team_list):
                potential_switch = team_list[switch_target_idx]
                # Check if the target is valid (in available_switches, not active, not fainted)
                if potential_switch in battle.available_switches and not potential_switch.active and not potential_switch.fainted:
                    return self.create_order(potential_switch)
                else:
                    # Fallback: Choose the first valid switch if the selected one is invalid
                    valid_switches = [p for p in battle.available_switches if p and not p.active and not p.fainted]
                    if valid_switches:
                        print(f"Warning: Action {action_idx} selected invalid switch target. Switching to {valid_switches[0].species}")
                        return self.create_order(valid_switches[0])
                    else:
                        # If no valid switches, must use a move (or default if no moves either)
                        print(f"Warning: Action {action_idx} selected switch, but no valid switches available. Choosing default move.")
                        return self.choose_default_move() # Pass battle object
            else:
                 # Fallback if index is out of team bounds (shouldn't happen with 6 max team size)
                 print(f"Warning: Action {action_idx} resulted in invalid team index {switch_target_idx}. Choosing default move.")
                 return self.choose_default_move() # Pass battle object

        # Should not happen with ACTION_DIM=10
        else:
            print(f"Error: Action index {action_idx} is out of range [0, 9]. Choosing default move.")
            return self.choose_default_move() # Pass battle object

    # --- Action Selection (Masking Refined & Robust Fallback) ---
    def select_action(self, state_tensor: torch.Tensor, battle: Battle) -> int:
        sample = random.random()
        eps_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                        np.exp(-1. * self.steps_done / EPSILON_DECAY)
        # Note: steps_done is incremented in choose_move now

        available_moves = battle.available_moves
        available_switches = battle.available_switches # These are Pokemon objects

        # Create a mask for valid actions
        valid_action_mask_np = np.zeros(ACTION_DIM, dtype=bool)
        possible_action_indices = []

        # Check valid moves (indices 0-3)
        can_move = False
        if not battle.force_switch: # Can only move if not forced to switch
             for i in range(len(available_moves)):
                 move = available_moves[i]
                 if i < 4: # Ensure we only consider the first 4 slots for moves
                     # Check if move is selectable (not disabled, sufficient PP unless struggling)
                     if move.current_pp > 0:
                         valid_action_mask_np[i] = True
                         possible_action_indices.append(i)
                         can_move = True
                     elif move.id == 'struggle': # Explicitly allow struggle if it appears
                         valid_action_mask_np[i] = True
                         possible_action_indices.append(i)
                         can_move = True

             # If no moves have PP > 0, and struggle isn't listed, but we aren't forced to switch,
             # the game usually forces Struggle. We mark the first move slot as valid
             # to allow the agent to select it (it will likely become Struggle).
             if not can_move and available_moves and not battle.force_switch:
                  # Check if struggle is implicitly available (no PP moves)
                  if all(m.current_pp == 0 for m in available_moves):
                       print("Warning: No moves with PP, assuming Struggle is possible via action 0.")
                       valid_action_mask_np[0] = True
                       if 0 not in possible_action_indices: possible_action_indices.append(0)
                       can_move = True # We can technically 'move' (Struggle)


        # Check valid switches (indices 4-9)
        can_switch = False
        if battle.available_switches: # Check if switches are available at all
            team_list = list(battle.team.values())
            for i in range(len(team_list)):
                pokemon = team_list[i]
                # Use a different variable name inside the loop to avoid potential confusion
                current_action_idx = i + 4 # Map team index (0-5) to action index (4-9)
                # Check if pokemon is actually in the available_switches list provided by battle object
                # and ensure it's a valid target (not active, not fainted)
                if current_action_idx < ACTION_DIM and pokemon in available_switches and not pokemon.active and not pokemon.fainted:
                    valid_action_mask_np[current_action_idx] = True
                    if current_action_idx not in possible_action_indices: possible_action_indices.append(current_action_idx)
                    can_switch = True

        # ***** CORRECTED INDENTATION *****
        # --- Handle Cases Where No Actions Seem Possible ---
        if not possible_action_indices:
            print(f"CRITICAL WARNING: No valid actions found by masking logic! "
                  f"force_switch={battle.force_switch}, "
                  f"moves={len(battle.available_moves)}, "
                  f"switches={len(battle.available_switches)}. "
                  f"Attempting default choice.")
            # If absolutely nothing seems valid, ask poke-env for its default
            default_choice = self.choose_default_move() # Get the default Order object
            if default_choice:
                # Check if it's a move order
                if hasattr(default_choice, 'move') and default_choice.move:
                     try:
                         # Find the index of the default move in the original list
                         # Need to handle case where available_moves might be empty
                         if battle.available_moves:
                             move_idx = battle.available_moves.index(default_choice.move)
                             if move_idx < 4:
                                 print(f"Using default move index: {move_idx}")
                                 return move_idx
                         else: # If no moves were available, default move must be struggle (action 0)
                             print("No available moves, default must be Struggle. Returning action 0.")
                             return 0
                     except (ValueError, IndexError): pass # Move not found or list empty? Fallback below.
                # Check if it's a switch order
                elif hasattr(default_choice, 'pokemon') and default_choice.pokemon:
                     try:
                         # Find the index of the default switch in the team list
                         team_idx = list(battle.team.values()).index(default_choice.pokemon)
                         action_idx_default = team_idx + 4 # Use different name
                         if action_idx_default < ACTION_DIM:
                             print(f"Using default switch index: {action_idx_default}")
                             return action_idx_default
                     except ValueError: pass # Pokemon not found? Fallback below.

            # Absolute fallback if default doesn't map cleanly or isn't available
            print("Default choice failed, unmappable, or not a move/switch. Returning action index 0 as last resort.")
            return 0 # Return index 0 if default fails


        # Epsilon-greedy selection (This section only runs if possible_action_indices was NOT empty)
        # Initialize action_idx before the if/else to ensure it's always assigned if this point is reached
        action_idx = -1 # Initialize with an invalid value

        if sample < eps_threshold:
            # Choose a random *valid* action
            action_idx = random.choice(possible_action_indices)
            # print(f"Exploring: Chose action {action_idx} from {possible_action_indices}")

        else:
            # Choose the best *valid* action according to the policy net
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)[0] # Get Q-values for the single state
                # Apply mask: set Q-values of invalid actions to negative infinity
                masked_q_values = torch.where(torch.tensor(valid_action_mask_np, device=device),
                                              q_values,
                                              torch.tensor(-float('inf'), device=device))

                # Check if all masked values are -inf (shouldn't happen if possible_action_indices is not empty)
                if torch.all(masked_q_values == -float('inf')):
                     print(f"CRITICAL WARNING: All actions masked to -inf despite possible actions {possible_action_indices}. Choosing random valid.")
                     action_idx = random.choice(possible_action_indices)
                else:
                     action_idx = masked_q_values.argmax().item()
                     # print(f"Exploiting: Chose action {action_idx} (Q: {masked_q_values[action_idx]:.2f}) from masked Qs: {masked_q_values}")
                     # Sanity check: Ensure the chosen action is actually valid
                     if not valid_action_mask_np[action_idx]:
                         print(f"CRITICAL ERROR: Argmax selected invalid action {action_idx}! Q-vals: {q_values}, Masked Q-vals: {masked_q_values}. Choosing random valid.")
                         action_idx = random.choice(possible_action_indices)

        # Final return (action_idx should always be assigned if this point is reached)
        if action_idx == -1:
             # This should be impossible if possible_action_indices was not empty
             print("CRITICAL ERROR: action_idx was not assigned in epsilon-greedy block! Falling back to random.")
             # Check possible_action_indices again just in case, though it shouldn't be empty here
             action_idx = random.choice(possible_action_indices) if possible_action_indices else 0

        return action_idx

    # --- Reward Calculation ---
    def _calculate_potential(self, battle: Battle) -> float:
        """Calculates a potential value based on the current battle state."""
        potential = 0.0

        # HP difference component
        own_hp = sum(p.current_hp_fraction for p in battle.team.values() if not p.fainted)
        opp_hp = sum(p.current_hp_fraction for p in battle.opponent_team.values() if not p.fainted)
        potential += (own_hp - opp_hp) * self.hp_potential_weight

        # Fainted Pokemon difference component
        own_fainted = sum(1 for p in battle.team.values() if p.fainted)
        opp_fainted = sum(1 for p in battle.opponent_team.values() if p.fainted)
        potential += (opp_fainted - own_fainted) * self.fainted_potential_weight

        # Status component (optional, can be noisy)
        # own_status = sum(1 for p in battle.team.values() if p.status is not None and not p.fainted)
        # opp_status = sum(1 for p in battle.opponent_team.values() if p.status is not None and not p.fainted)
        # potential -= (own_status - opp_status) * self.status_potential_weight

        return potential

    def _create_battle_snapshot(self, battle: Battle) -> dict:
        """Creates a dictionary snapshot of key battle elements for reward comparison."""
        return {
            "own_fainted": sum(1 for p in battle.team.values() if p.fainted),
            "opp_fainted": sum(1 for p in battle.opponent_team.values() if p.fainted),
            # Add other elements if needed for event rewards (e.g., hazard counts)
        }

    def compute_reward(self, battle: Battle) -> float:
        """Computes the reward based on state changes and battle outcome."""
        reward = 0.0
        current_potential = self._calculate_potential(battle)
        current_snapshot = self._create_battle_snapshot(battle)

        # 1. Win/Loss Reward (Terminal states)
        if battle.won:
            reward += self.win_reward
        elif battle.lost:
            reward += self.lose_penalty

        # 2. Potential Shaping Reward (Non-terminal states)
        # Only apply if we have a previous potential value (i.e., not the first turn)
        if self.previous_battle_snapshot is not None:
            reward += (GAMMA * current_potential - self.last_potential)

        # 3. Event-Based Rewards (Comparing current state to previous snapshot)
        if self.previous_battle_snapshot is not None:
            # Opponent Pokemon fainted
            opp_fainted_diff = current_snapshot["opp_fainted"] - self.previous_battle_snapshot["opp_fainted"]
            if opp_fainted_diff > 0:
                reward += opp_fainted_diff * self.ko_reward
                # print(f"Reward: +{opp_fainted_diff * self.ko_reward} for KO")


            # Own Pokemon fainted
            own_fainted_diff = current_snapshot["own_fainted"] - self.previous_battle_snapshot["own_fainted"]
            if own_fainted_diff > 0:
                reward -= own_fainted_diff * self.faint_penalty
                # print(f"Reward: -{own_fainted_diff * self.faint_penalty} for Faint")


            # Add other event rewards here (e.g., setting/removing hazards, status infliction)
            # Requires adding relevant info to _create_battle_snapshot

        # Update state for the *next* reward calculation
        self.last_potential = current_potential
        self.previous_battle_snapshot = current_snapshot # Store the *current* snapshot for the next step

        # Clip reward? Optional, can help stabilize if rewards explode
        # reward = max(-1.0, min(reward, 1.0)) # Example clipping

        return reward

    # --- Learning Step ---
    def learn(self):
        if not self.memory.is_ready(BATCH_SIZE):
            return 0.0 # Return 0 loss if not ready

        experiences = self.memory.sample(BATCH_SIZE)
        if not experiences: # Should not happen if is_ready is checked, but safety first
             return 0.0

        batch = Experience(*zip(*experiences))

        # Convert numpy arrays stored in memory to tensors for the batch
        state_batch = torch.stack([torch.from_numpy(s) for s in batch.state]).to(device)
        action_batch = torch.tensor(batch.action, device=device, dtype=torch.long).unsqueeze(1) # Ensure long type for gather
        reward_batch = torch.tensor(batch.reward, device=device, dtype=torch.float32)
        next_state_batch = torch.stack([torch.from_numpy(s) for s in batch.next_state]).to(device)
        done_batch = torch.tensor(batch.done, device=device, dtype=torch.bool)


        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # V(s) = max_a Q_target(s, a)
        next_state_values = torch.zeros(len(experiences), device=device) # Use len(experiences) which is actual batch size
        # We only want to compute values for states that are not terminal
        non_final_mask = ~done_batch
        non_final_next_states = next_state_batch[non_final_mask]

        # Ensure there are non-final states before accessing target_net
        if non_final_next_states.size(0) > 0:
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values: R + gamma * V(s_{t+1})
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss (Smooth L1)
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping (optional but recommended for stability)
        # torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100) # Clip gradients example
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0) # Clip norm example
        self.optimizer.step()

        # Update target network (using steps_done which increments each turn)
        if self.steps_done % TARGET_UPDATE_FREQ == 0:
            # print(f"Updating target network at step {self.steps_done}")
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    # --- Main Battle Logic ---
    def choose_move(self, battle: Battle):
        # Increment steps_done at the beginning of choose_move call
        self.steps_done += 1

        current_state_vec = self.embed_battle(battle)
        current_state_tensor = torch.from_numpy(current_state_vec).unsqueeze(0).to(device)
        calculated_reward = 0.0 # Default reward if it's the first turn

        # --- Reward Calculation and Memory Push ---
        # Check if this is not the first action of the battle
        if self.last_battle_state_vec is not None and self.last_action_idx is not None:
            # Calculate reward for the transition from the *previous* state to the *current* one
            calculated_reward = self.compute_reward(battle) # This updates self.last_potential and self.previous_battle_snapshot

            # Store the experience from the *previous* step
            self.memory.push(self.last_battle_state_vec, self.last_action_idx, calculated_reward, current_state_vec, battle.finished)

            # Learn from memory
            loss = self.learn()
            # if self.steps_done % 100 == 0: # Log loss periodically
            #      print(f"Step: {self.steps_done}, Loss: {loss:.4f}, Reward: {calculated_reward:.3f}")


        # --- Handle First Turn ---
        elif battle.turn == 1 or self.previous_battle_snapshot is None:
            # Initialize potential and snapshot for the very first state observed
            self.last_potential = self._calculate_potential(battle)
            self.previous_battle_snapshot = self._create_battle_snapshot(battle)
            # print("Initialized potential and snapshot for new battle.")

        # --- Action Selection ---
        action_idx = self.select_action(current_state_tensor, battle)

        # --- Update State for Next Step ---
        if battle.finished:
            # Reset state tracking for the next battle
            # print(f"Battle Finished. Won: {battle.won}. Final Reward Step: {calculated_reward:.3f}")
            self.last_battle_state_vec = None
            self.last_action_idx = None
            self.last_potential = 0.0
            self.previous_battle_snapshot = None
        else:
            # Store the current state and action for the *next* reward calculation
            self.last_battle_state_vec = current_state_vec
            self.last_action_idx = action_idx
            # Note: self.last_potential and self.previous_battle_snapshot were already updated
            # within compute_reward if it was called, or initialized on turn 1.

        # --- Convert action to order ---
        order = self._action_to_move(action_idx, battle)
        return order

    # --- Model Saving ---
    def save_model(self, path=None):
        if path is None:
            path = MODEL_PATH
        print(f"Saving model to {path}...")
        torch.save(self.policy_net.state_dict(), path)
        print("Model saved.")

    # --- Model Loading ---
    def load_model(self, path=None):
         if path is None:
             path = MODEL_PATH
         try:
             print(f"Loading model from {path}...")
             # Set weights_only=True for added security if loading untrusted files
             # You might need to set weights_only=False if your model file was saved
             # with an older PyTorch version or contains non-tensor data.
             try:
                 state_dict = torch.load(path, map_location=device, weights_only=True)
             except Exception: # Fallback if weights_only=True fails
                 print("Warning: Loading with weights_only=True failed. Attempting weights_only=False.")
                 state_dict = torch.load(path, map_location=device, weights_only=False)

             self.policy_net.load_state_dict(state_dict)
             self.target_net.load_state_dict(self.policy_net.state_dict()) # Sync target net
             self.policy_net.eval() # Set to evaluation mode if not training further
             self.target_net.eval()
             print("Model loaded successfully.")
         except FileNotFoundError:
             print(f"Info: Model file not found at {path}. Starting with random weights.")
         except Exception as e:
             print(f"Error loading model: {e}. Starting with random weights.")