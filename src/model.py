import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from poke_env.player import Player
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.move import Move
from poke_env.environment.pokemon import Pokemon


# --- Environment Class ---
# Primarily provides helper functions now
class PokemonEnvironment:
    def __init__(self):
        # Constants definitions (Types, Status, etc.)
        self.POKEMON_TYPES = ["normal", "fire", "water", "electric", "grass", "ice", "fighting", "poison",
                              "ground", "flying", "psychic", "bug", "rock", "ghost", "dragon", "dark", "steel", "fairy"] # Added Fairy
        self.TYPE_TO_IDX = {t: i for i, t in enumerate(self.POKEMON_TYPES)}
        self.NUM_TYPES = len(self.POKEMON_TYPES) # 18

        self.STATUS_CONDITIONS = ["psn", "tox", "par", "slp", "frz", "brn"]
        self.STATUS_TO_IDX = {s: i for i, s in enumerate(self.STATUS_CONDITIONS)}
        self.NUM_STATUS = len(self.STATUS_CONDITIONS) # 6

        # Recalculate observation_dim based on embedding structure
        # Player types(18) + Opp types(18) + Player HP(1) + Opp HP(1) + Player Stat(6) + Opp Stat(6)
        # + 4 * [Move Power(1) + Move Acc(1) + Move Type(18)]
        # = 18 + 18 + 1 + 1 + 6 + 6 + 4 * (1 + 1 + 18)
        # = 50 + 4 * 20 = 50 + 80 = 130
        self.observation_dim = 130
        self.action_dim = 9 # 4 moves + 5 switches (Max 6 pokemon, 1 active -> 5 switches)

    # compute_reward remains the same
    def compute_reward(self, battle: AbstractBattle) -> float:
        return self.reward_computing_helper(
            battle,
            fainted_value=2.0,
            hp_value=1.0,
            victory_value=30.0
        )

    # reward_computing_helper remains the same
    def reward_computing_helper(self, battle: AbstractBattle, fainted_value: float = 2.0,
        hp_value: float = 1.0, victory_value: float = 30.0) -> float:

        if battle.ended:
            return victory_value * (1.0 if battle.won else -1.0)

        # Intermediate reward based on current state snapshot
        total_hp_reward = 0.0
        if battle.active_pokemon and battle.opponent_active_pokemon:
            total_hp_reward = hp_value * (battle.active_pokemon.current_hp_fraction - battle.opponent_active_pokemon.current_hp_fraction)
        elif battle.active_pokemon:
             total_hp_reward = hp_value * battle.active_pokemon.current_hp_fraction
        elif battle.opponent_active_pokemon:
             total_hp_reward = -hp_value * battle.opponent_active_pokemon.current_hp_fraction

        fainted_opponent = len([p for p in battle.opponent_team.values() if p.fainted])
        fainted_self = len([p for p in battle.team.values() if p.fainted])
        fainted_reward = fainted_value * (fainted_opponent - fainted_self)

        # Consider adding penalties for self-status or rewards for opponent status later

        return total_hp_reward + fainted_reward

    # embed_battle needs careful index checking
    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        embedding = np.zeros(self.observation_dim, dtype=np.float32)
        offset = 0

        # Player active Pokemon types (Size: NUM_TYPES) Indices 0..NUM_TYPES-1
        if battle.active_pokemon:
            for p_type in battle.active_pokemon.types:
                if p_type:
                    type_str = p_type.name.lower()
                    if type_str in self.TYPE_TO_IDX:
                        embedding[offset + self.TYPE_TO_IDX[type_str]] = 1.0
        offset += self.NUM_TYPES

        # Opponent active Pokemon types (Size: NUM_TYPES) Indices NUM_TYPES..2*NUM_TYPES-1
        if battle.opponent_active_pokemon:
            for p_type in battle.opponent_active_pokemon.types:
                 if p_type:
                    type_str = p_type.name.lower()
                    if type_str in self.TYPE_TO_IDX:
                        embedding[offset + self.TYPE_TO_IDX[type_str]] = 1.0
        offset += self.NUM_TYPES

        # Player HP fraction (Size: 1)
        if battle.active_pokemon:
            embedding[offset] = battle.active_pokemon.current_hp_fraction
        offset += 1

        # Opponent HP fraction (Size: 1)
        if battle.opponent_active_pokemon:
            embedding[offset] = battle.opponent_active_pokemon.current_hp_fraction
        offset += 1

        # Player Status (Size: NUM_STATUS)
        if battle.active_pokemon and battle.active_pokemon.status:
            status_str = battle.active_pokemon.status.name.lower()
            if status_str in self.STATUS_TO_IDX:
                embedding[offset + self.STATUS_TO_IDX[status_str]] = 1.0
        offset += self.NUM_STATUS

        # Opponent Status (Size: NUM_STATUS)
        if battle.opponent_active_pokemon and battle.opponent_active_pokemon.status:
            status_str = battle.opponent_active_pokemon.status.name.lower()
            if status_str in self.STATUS_TO_IDX:
                embedding[offset + self.STATUS_TO_IDX[status_str]] = 1.0
        offset += self.NUM_STATUS

        # Available Moves (4 moves max)
        # Each move: Power (1), Accuracy (1), Type (NUM_TYPES) -> Total 2 + NUM_TYPES per move
        move_info_size = 2 + self.NUM_TYPES # Should be 20
        if battle.available_moves:
             for i, move in enumerate(battle.available_moves):
                 if i >= 4: break
                 move_base_idx = offset + i * move_info_size

                 # Power (Normalized by 150)
                 embedding[move_base_idx] = min(move.base_power / 150.0, 1.0) if move.base_power > 0 else 0.0

                 # Accuracy (Normalized)
                 if isinstance(move.accuracy, (int, float)):
                     embedding[move_base_idx + 1] = move.accuracy / 100.0
                 elif move.accuracy is True: # Always hits
                     embedding[move_base_idx + 1] = 1.0
                 else: # None accuracy (e.g., status moves that usually hit)
                     embedding[move_base_idx + 1] = 1.0 # Treat as perfect

                 # Type (One-hot)
                 if move.type:
                     type_str = move.type.name.lower()
                     if type_str in self.TYPE_TO_IDX:
                         embedding[move_base_idx + 2 + self.TYPE_TO_IDX[type_str]] = 1.0
        # Final check: offset should be 50 here.
        # The embedding size is 50 + 4 * move_info_size = 50 + 4 * 20 = 130. Correct.

        return embedding

    # action_to_move needs robust checking
    def action_to_move(self, battle: AbstractBattle, action_idx: int) -> Move | Pokemon | None:
        available_moves = battle.available_moves
        available_switches = battle.available_switches

        # --- Check for forced actions first ---
        if battle.force_switch: # Must switch
            if not available_switches:
                 print(f"[ERROR] Forced switch but no switches available! Battle: {battle.battle_tag}")
                 # This shouldn't happen if Pokemon are available, maybe return default/random valid?
                 # For now, return None, let choose_move handle fallback.
                 return None
            # If forced switch, actions 0-3 are invalid. Map 4-8 to switches.
            if action_idx < 4: # Agent chose a move, but must switch
                 print(f"[Warning] action_to_move: Agent chose move ({action_idx}) but force_switch=True. Choosing random switch.")
                 return random.choice(available_switches)
            else: # Agent chose a switch action (4-8)
                 switch_idx = action_idx - 4
                 if switch_idx < len(available_switches):
                     return available_switches[switch_idx]
                 else:
                     print(f"[Warning] action_to_move: Invalid switch index {switch_idx} during force_switch. Choosing random switch.")
                     return random.choice(available_switches)

        if battle.trapped: # Cannot switch
             if not available_moves:
                  print(f"[ERROR] Trapped but no moves available! Battle: {battle.battle_tag}")
                  # Should have Struggle, implies state error. Return None for fallback.
                  return None
             # If trapped, actions 4-8 are invalid. Map 0-3 to moves.
             if action_idx >= 4: # Agent chose a switch, but trapped
                  print(f"[Warning] action_to_move: Agent chose switch ({action_idx}) but trapped=True. Choosing random move.")
                  return random.choice(available_moves)
             else: # Agent chose a move action (0-3)
                  if action_idx < len(available_moves):
                      return available_moves[action_idx]
                  else:
                      print(f"[Warning] action_to_move: Invalid move index {action_idx} while trapped. Choosing random move.")
                      return random.choice(available_moves)

        # --- Standard case: Not forced switch or trapped ---
        # Action 0-3: Moves
        if action_idx < 4:
            if not available_moves: # No moves available (e.g., choice locked, recharging) but not trapped? Rare.
                 print(f"[Warning] action_to_move: Chose move ({action_idx}) but no moves available (and not trapped/forced). Trying switches.")
                 if available_switches:
                     return random.choice(available_switches)
                 else:
                     print(f"[ERROR] action_to_move: No moves or switches available in standard case! Battle: {battle.battle_tag}")
                     return None # Let choose_move handle fallback
            elif action_idx < len(available_moves):
                return available_moves[action_idx]
            else:
                # Requested move index is invalid (e.g., agent chose action 2, but only 1 move available)
                print(f"[Warning] action_to_move: Invalid move index {action_idx} ({len(available_moves)} available). Choosing random move.")
                return random.choice(available_moves)

        # Action 4-8: Switches
        else:
            if not available_switches: # No switches available (e.g., only 1 pokemon left)
                print(f"[Warning] action_to_move: Chose switch ({action_idx}) but no switches available. Trying moves.")
                if available_moves:
                    return random.choice(available_moves)
                else:
                    print(f"[ERROR] action_to_move: No moves or switches available in standard case! Battle: {battle.battle_tag}")
                    return None # Let choose_move handle fallback
            else:
                switch_idx = action_idx - 4 # Convert action index (4-8) to switch list index (0-4)
                if switch_idx < len(available_switches):
                    return available_switches[switch_idx]
                else:
                    # Requested switch index is invalid
                    print(f"[Warning] action_to_move: Invalid switch index {switch_idx} ({len(available_switches)} available). Choosing random switch.")
                    return random.choice(available_switches)

# --- DQNAgent Class ---
# --- DQNAgent Class ---
class DQNAgent(Player):
    def __init__(self,
                 battle_format="gen4randombattle",
                 input_dim=130, # Updated based on environment
                 output_dim=9,  # Updated based on environment
                 epsilon=1.0,
                 epsilon_min=0.05, # Lower min epsilon
                 epsilon_decay=0.9995, # Slower decay? Adjust based on steps/battle
                 gamma=0.97, # Slightly higher discount factor
                 learning_rate=0.00025, # Common starting LR for Adam
                 batch_size=64, # Larger batch size
                 memory_size=50000, # Increased memory
                 target_update_frequency=1000, # Update target net less frequently
                 model_path="models/pokemon_dqn_v2.pth"): # New model path
        super().__init__(battle_format=battle_format, max_concurrent_battles=10) # Allow more battles

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.model_path = model_path
        self.target_update_frequency = target_update_frequency

        # Use a single instance of the helper environment
        self.env = PokemonEnvironment()

        # --- State Tracking Initialized EARLY ---
        self.memory = [] # Initialize memory (or use a deque)
        self.current_battle_states = {}
        self.current_battle_actions = {}
        self.training_steps = 0
        self.update_counter = 0
        self.battle_count = 0
        self.losses = []
        self.battle_results = []
        # --- End Early Initialization ---

        # Now build models and optimizer
        self.model = self.build_model(input_dim, output_dim)
        self.target_model = self.build_model(input_dim, output_dim)
        self.update_target_model() # Initialize target model (Now it's safe)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        # No need for self.is_training = False

    # ... rest of the DQNAgent class remains the same ...

    def build_model(self, input_dim, output_dim):
        # Consider BatchNorm or Dropout if overfitting becomes an issue later
        model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256), # Added another layer
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        # print("Model Architecture:") # Only print once if desired
        # print(model)
        return model

    def update_target_model(self):
        # This print statement is now safe because self.training_steps exists
        print(f"Updating target model at step {self.training_steps}")
        self.target_model.load_state_dict(self.model.state_dict())

    # ... rest of the DQNAgent class ...

    # This method IS CALLED BY POKE-ENV when a move is needed
    def choose_move(self, battle: AbstractBattle):
        battle_tag = battle.battle_tag

        # --- Step 1: Record Experience from *Previous* Action (if any) ---
        if battle_tag in self.current_battle_states:
            last_state = self.current_battle_states[battle_tag]
            last_action = self.current_battle_actions[battle_tag]
            reward = self.env.compute_reward(battle) # Reward based on state AFTER action/opponent move
            next_state = self.env.embed_battle(battle) # Current state is the 'next_state' for the last action
            done = battle.ended

            self.remember(last_state, last_action, reward, next_state, done)

            # --- Step 2: Perform Learning (Replay) ---
            if len(self.memory) >= self.batch_size:
                loss = self.replay()
                if loss is not None: # replay returns None if not enough memory
                    self.training_steps += 1
                    # Target model update logic moved inside replay()

        # --- Step 3: Choose Action for the *Current* State ---
        current_state_embedding = self.env.embed_battle(battle)
        state_tensor = torch.tensor(np.reshape(current_state_embedding, [1, self.input_dim]), dtype=torch.float32)

        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            action_idx = random.randrange(self.output_dim)
            # Optional: Try to pick a valid random action first
            valid_action_indices = self.get_valid_action_indices(battle)
            if valid_action_indices:
                 action_idx = random.choice(valid_action_indices)
            # else: use the totally random index, action_to_move will handle it
        else:
            with torch.no_grad():
                q_values = self.model(state_tensor)
                # Optional but recommended: Mask invalid actions before argmax
                q_values = self.mask_invalid_actions(battle, q_values)
            action_idx = torch.argmax(q_values[0]).item()

        # --- Step 4: Map Action Index to Move/Pokemon ---
        # Use the environment helper function
        # This function now contains robust checks for forced actions, traps, etc.
        move_or_switch = self.env.action_to_move(battle, action_idx)

        # --- Step 5: Fallback if action mapping failed ---
        if move_or_switch is None:
             print(f"[CRITICAL FALLBACK] choose_move: action_to_move failed for action {action_idx} in battle {battle_tag}. Choosing random valid action.")
             # Use poke-env's built-in random choice as a safe fallback
             move_or_switch = self.choose_random_move(battle)
             # We might want to map this back to an action index if possible for storage, but it's complex.
             # For now, store the original failed action index.
             # Consider penalizing the Q-value for the failed action index later.


        # --- Step 6: Store State and Action for Next Learning Step ---
        self.current_battle_states[battle_tag] = current_state_embedding
        self.current_battle_actions[battle_tag] = action_idx # Store the chosen *index*

        # Debug print
        # print(f"Turn {battle.turn}: Player {battle.player_role} chose action {action_idx} -> {type(move_or_switch).__name__}: {getattr(move_or_switch, 'id', getattr(move_or_switch, 'species', 'Unknown'))}")

        # --- Step 7: Return the chosen Move/Pokemon object ---
        # poke-env handles wrapping this in BattleOrder and sending the command
        return move_or_switch


    def get_valid_action_indices(self, battle: AbstractBattle) -> list[int]:
        """Gets a list of action indices that correspond to currently valid moves/switches."""
        indices = []
        # Moves (0-3)
        for i in range(len(battle.available_moves)):
            indices.append(i)
        # Switches (4-8)
        for i in range(len(battle.available_switches)):
            indices.append(i + 4)

        # Handle forced actions explicitly (override previous)
        if battle.force_switch:
            indices = [i + 4 for i in range(len(battle.available_switches))]
        elif battle.trapped:
             indices = [i for i in range(len(battle.available_moves))]

        # Ensure indices are within the defined action space dim
        indices = [idx for idx in indices if 0 <= idx < self.output_dim]

        # If no actions possible (should only happen with Struggle or error)
        if not indices and battle.available_moves: # Check if struggle exists?
             # Assume index 0 corresponds to the first move (potentially struggle)
             indices = [0]

        return indices

    def mask_invalid_actions(self, battle: AbstractBattle, q_values: torch.Tensor) -> torch.Tensor:
        """Sets Q-values of invalid actions to a very low number."""
        valid_indices = self.get_valid_action_indices(battle)
        mask = torch.full_like(q_values, -float('inf')) # Mask with negative infinity
        if valid_indices: # Check if list is not empty
             mask[0, valid_indices] = 0 # Set valid actions mask to 0
        else:
             # If no valid actions, maybe don't mask? Or mask all except first?
             # This case is tricky, implies potentially only struggle or error state.
             # Let's allow all actions if none are explicitly valid, argmax will pick one.
             # action_to_move / choose_move fallback should handle it.
             # Alternatively, mask all except action 0 (assuming it maps to first move/struggle)
             # mask[0, 0] = 0
             pass # Don't mask if no valid indices found, let argmax proceed

        return q_values + mask # Add mask (valid actions keep original q, invalid become -inf)


    # Override _battle_finished_callback to clean up state and record final step
    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        battle_tag = battle.battle_tag
        print(f"Battle {battle_tag} finished. Won: {battle.won}. Epsilon: {self.epsilon:.4f}")

        # --- Record Final Experience ---
        if battle_tag in self.current_battle_states:
            last_state = self.current_battle_states[battle_tag]
            last_action = self.current_battle_actions[battle_tag]
            # Use final battle state for reward and next_state
            final_reward = self.env.compute_reward(battle)
            final_state = self.env.embed_battle(battle)
            self.remember(last_state, last_action, final_reward, final_state, True) # Done = True

            # Optional: One last replay step after the final experience
            if len(self.memory) >= self.batch_size:
                 loss = self.replay()
                 # if loss is not None: self.training_steps += 1 # Already counted in replay

            # Clean up state tracking for this battle
            del self.current_battle_states[battle_tag]
            del self.current_battle_actions[battle_tag]

        # --- Update Battle Stats ---
        self.battle_count += 1
        self.battle_results.append(1 if battle.won else 0)

        # Epsilon decay moved to replay() to happen per learning step
        # Target network update moved to replay()


    def remember(self, state, action, reward, next_state, done):
        """Stores experience in replay memory."""
        if len(self.memory) < self.memory_size:
            self.memory.append(None) # Pre-allocate? Or just append
        # Use modulo for circular buffer
        index = self.training_steps % self.memory_size # Use training_steps to track position
        self.memory[index] = (state, action, reward, next_state, done)
        # self.memory.append((state, action, reward, next_state, done))
        # if len(self.memory) > self.memory_size:
        #     self.memory.pop(0) # Simple deque is fine too


    def replay(self):
        """Samples from memory and performs a gradient descent step."""
        if len(self.memory) < self.batch_size:
            return None # Not enough memory yet

        # Sample a minibatch (filter out None if using pre-allocation)
        valid_memory = [exp for exp in self.memory if exp is not None]
        if len(valid_memory) < self.batch_size:
            return None
        minibatch = random.sample(valid_memory, self.batch_size)

        # --- Prepare Tensors ---
        states = torch.tensor(np.array([exp[0] for exp in minibatch]), dtype=torch.float32)
        actions = torch.tensor([exp[1] for exp in minibatch], dtype=torch.long)
        rewards = torch.tensor([exp[2] for exp in minibatch], dtype=torch.float32)
        next_states = torch.tensor(np.array([exp[3] for exp in minibatch]), dtype=torch.float32)
        dones = torch.tensor([exp[4] for exp in minibatch], dtype=torch.float32) # Ensure float 0.0 or 1.0

        # --- Calculate Target Q-values ---
        with torch.no_grad():
            # Use target network for next state Q values
            next_q_values = self.target_model(next_states)
            # Select best action according to policy network (Double DQN)
            best_actions = self.model(next_states).argmax(dim=1)
            # Get Q value from target network corresponding to policy network's best action
            max_next_q_values = next_q_values.gather(1, best_actions.unsqueeze(1)).squeeze(1)
            # If done, target is just reward. Otherwise, reward + discounted future Q.
            target_q_values = rewards + (1.0 - dones) * self.gamma * max_next_q_values

        # --- Calculate Current Q-values ---
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # --- Calculate Loss ---
        loss = self.criterion(current_q_values, target_q_values)

        # --- Optimize Model ---
        self.optimizer.zero_grad()
        loss.backward()
        # Optional: Gradient Clipping
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # --- Epsilon Decay (per learning step) ---
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # --- Target Network Update ---
        self.update_counter += 1
        if self.update_counter % self.target_update_frequency == 0:
            self.update_target_model() # Update target network

        loss_value = loss.item()
        self.losses.append(loss_value) # Store loss for monitoring
        return loss_value


    def save_model(self, path=None):
        if path is None:
            path = self.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Save more comprehensive state
        save_data = {
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'update_counter': self.update_counter,
            'battle_count': self.battle_count,
            'battle_results': self.battle_results,
            'losses': self.losses,
            'memory': self.memory # Saving memory can be large! Optional.
        }
        # Limit size of saved losses/results if needed
        max_len = 10000
        save_data['battle_results'] = save_data['battle_results'][-max_len:]
        save_data['losses'] = save_data['losses'][-max_len:]

        try:
             torch.save(save_data, path)
             print(f"Model and state saved to {path}")
        except Exception as e:
             print(f"Error saving model to {path}: {e}")


    def load_model(self, path=None):
        if path is None:
            path = self.model_path
        if os.path.exists(path):
            try:
                checkpoint = torch.load(path)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epsilon = checkpoint.get('epsilon', self.epsilon)
                self.training_steps = checkpoint.get('training_steps', 0)
                self.update_counter = checkpoint.get('update_counter', 0)
                self.battle_count = checkpoint.get('battle_count', 0)
                self.battle_results = checkpoint.get('battle_results', [])
                self.losses = checkpoint.get('losses', [])
                # Load memory if saved and desired (can take time/RAM)
                if 'memory' in checkpoint:
                     self.memory = checkpoint['memory']
                     print(f"Loaded {len(self.memory)} experiences from memory.")

                self.model.train() # Ensure model is in training mode
                self.target_model.eval() # Target model is usually in eval mode
                print(f"Model and state loaded from {path}")
                print(f"Resuming training from step {self.training_steps}, battle {self.battle_count+1}, epsilon {self.epsilon:.4f}")

            except Exception as e:
                 print(f"Error loading model from {path}: {e}. Starting fresh.")
                 self.update_target_model() # Ensure target matches initial model
        else:
            print(f"No model found at {path}. Starting fresh.")
            self.update_target_model() # Ensure target matches initial model