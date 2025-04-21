# [rlmon](https://github.com/mohit065/rlmon) - A Pokémon Showdown Bot Using Reinforcement Learning

## Index

- [Overview](#overview)
- [Setting the Game](#setting-the-game)
- [Modeling the MDP](#modeling-the-mdp)
- [Q-Learning](#q-learning)
- [Results](#results)
- [Observations](#observations)
- [Challenges](#challenges)
- [Steps to Run](#steps-to-run)
- [Contributions](#contributions)

---

## Overview

This project aims to implement, train and evaluate a reinforcement learning agent to play Pokémon battles on the Pokémon Showdown battle simulator. We first formulate a Pokémon battle as a *Markov Decision Process* (MDP), then train the agent to learn the optimal policy at each state via Deep Q-Learning, and evaluate its performance against a set of standard opponents.

---

## Setting the Game

A Pokémon battle between a player and an opponent is a turn-based game where both sides have 6 Pokémon each. At any point of time, exactly one Pokémon from each side is *active* i.e. out for battle. At each turn, both the player and the opponent can either use their active Pokémon to make a move, or switch out their active Pokémon. The objective of the game is to defeat all 6 of the opponent's Pokémon first.

Each Pokémon has a number of statistics associated with it:

- Type: Each Pokémon can have one or two types.
- Status Conditions: During the course of the battle, a Pokémon can get afflicted with one of several possible status conditions (sleep, paralysis, burn, frost, poison), each of which has different effects on the other statistics of the Pokémon.
- Moves: Each Pokémon has a set of 4 moves, each of which have their own statistics:
  
  - Base Power: The damaging power of a move. Moves with higher base power will cause higher damage to the opponent Pokémon.
  - Accuracy: The chance of a move hitting the opponent.
  - Move category: Moves are categorized as physical, special and status.
  - Priority: Some moves with higher priority always go first.
  - Type: Each move has exactly one type, chosen from the set of Pokémon types. Moves with types matching the Pokémon playing that move get an increase in base power. Moves of a certain type may have higher effect / lower effect / no effect on Pokémon of a certain type, which is computed using a type matchup table.
  - PP: The number of times you can use a single move
  - Secondary effect: Some moves may have a secondary effect like increasing the stats of the active Pokémon or inflicting a status condition on the opponent.

- HP: The hit-points of a Pokémon. A damaging move will reduce the hit-points, and the Pokémon is defeated when the HP falls to zero.
- Atk / SpAtk: The attacking power of a Pokémon, which influences the base power of its moves.
- Def / SpDef: The defensive power of a Pokémon which influences how much HP it loses when hit by moves.
- Speed: A Pokémon's speed. Moves of faster Pokémon will go first.
- Abilities, Items, Natures: Each Pokémon has an ability and a nature, and can hold an item. These may boost one of its stats, reduce those of the opponent etc.

Note: To reduce complexity and avoid errors, we have ignored some functionalities like weather conditions, secondary status conditions like confusion/infatuation/flinching, single type Pokémon, complex moves, items and abilities, IVs. Also, the battle format used is *gen4anythinggoes*, which automatically discards Mega Evolutions, Z-Moves, Dynamax/Galarian Pokémon, and Terrastalize. This ensures consistency and simplicity while keeping the core philosophy of a Pokémon battle intact.

We created 6 teams of 6 dual-type Pokémon each, with every Pokémon having an item, an ability, a nature, EVs to boost stats, and a set of 4 moves. The first four teams are of similar strength, with the 5th team being slightly stronger and the 6th being slightly weaker.

---

## Modeling the MDP

A Pokémon Showdown battle is a classic example of a reinforcement learning problem. The agent must make a sequence of decisions — selecting moves or switching Pokémon — with the objective of maximizing long-term reward in a dynamic, uncertain environment.

- Sequential Decision-Making: At each turn, the agent must choose the best move or switch, balancing short-term gains with long-term strategy.
- Long-Term Objective: The ultimate goal is to win the battle — a reward that can only be realized after a potentially long sequence of turns.
- Uncertainty: The environment is stochastic and partially observable:
  
  - Moves can miss or land critical hits.
  - Status effects like paralysis or sleep may not activate consistently.
  - The opponent’s full team and moveset are not known initially and are only revealed over time.

The battle can also be formulated as a Markov Decision Process (MDP), which consists of:

- State Space (S): Encodes the full battle configuration at any point.
- Action Space (A): All valid moves or switches the player can make.
- Transition Function (P(s, a, s')): Probability distribution over next states, given current state and action.
- Reward Function (R(s, a, s')): Scalar feedback signal indicating how good the transition was.
- Discount Factor (γ): Weighs future rewards against immediate ones.

The Markov property is satisfied by carefully designing the state to fully capture all relevant information at the current turn, so the next state depends only on the current one and the chosen action — not the full history.

### State Space

The state is encoded as a 277-dimensional vector, capturing information about both the player and the opponent:

- Player (151 features)
  
  - HP Fractions: For all 6 Pokémon
  - Types: 2 per Pokémon, encoded numerically
  - Moves: Type, base power, accuracy, and PP fraction for 4 moves × 6 Pokémon
  - Status Conditions: e.g., burn, paralysis
  - Base Stats: Normalized values for Atk, Def, SpA, SpD, Spe for all 6 Pokémon
  - Active Pokémon Index

- Opponent (126 features)
  
  - Same categories as the player, but only include observable or revealed information
  - Unseen features are filled with -1.0 placeholders
  - Only the active Pokémon's stat boosts are included (not full base stats)

### Action Space

At any point in the battle, the agent can choose from up to 9 discrete actions:

- 4 Moves: If the active Pokémon has PP left
- 5 Switches: To any of the other non-fainted Pokémon

The actual number of valid actions depends on the battle state (e.g., fainted Pokémon cannot be switched to, and moves with 0 PP cannot be used).

### Reward Function

The agent is guided by both sparse terminal rewards and dense intermediate rewards:

- Terminal Rewards:
  
  - Large positive reward on winning
  - Large negative reward on losing

- Intermediate Shaped Rewards:
  
  Positive reward for:
  
  - Knocking out opponent Pokémon
  - Dealing damage
  - Inflicting status conditions

  Negative reward for:
  
  - Losing your own Pokémon

Each component is weighted by configurable constants.

---

## Q-Learning

Q-learning is an off-policy value-based method that estimates the optimal action-value function (Q-function), which tells the agent the expected future reward of taking a certain action in a given state. Once trained, this function allows the agent to act optimally by selecting actions that maximize long-term rewards.

### Why Not Tabular Q-Learning?

Although tabular Q-learning is easy to implement and intuitive, it fails in complex environments like Pokémon Showdown for a few key reasons:

- Massive state space: A single battle includes hundreds of Pokémon, each with unique stats, moves, abilities, and dynamic battle conditions. Tabular Q-learning requires explicitly storing a Q-value for every possible state-action pair, which is computationally infeasible here. We chose Q-learning, rather than other methods like policy gradients, due to its simplicity and effectiveness in discrete action spaces like turn-based Pokémon battles.

- High-dimensional, partially observable input: Much of the information in Pokémon is structured and only partially observable (e.g., hidden moves, unseen opponents). This makes generalization across similar states critical—something tabular methods can’t do.

- Lack of abstraction: Tabular approaches treat each state as unique, with no ability to infer similarities or shared structure between them, unlike neural networks.

Because of these limitations, we use a Deep Q-Network (DQN) to approximate the Q-function.

### DQN Architecture Overview

Our DQN consists of a feedforward neural network that approximates the Q-function. It contains:

- Input Layer: The state of the battle is encoded as a fixed-size numerical vector that captures detailed information about the player's and opponent's teams (HP, types, statuses, stats, move info, etc.). This embedding allows the network to generalize across similar game states.

- Hidden Layers: Two fully connected layers with ReLU activation functions allow the model to learn non-linear patterns and dependencies within the input state.

- Output Layer: A fixed-size vector representing Q-values for each possible action (moves and switches). The action with the highest Q-value is selected, unless exploring.

### Features

- Experience Replay: Transitions are stored in a replay buffer and sampled in batches during training. This breaks temporal correlations and improves stability.
  
- Target Network: A separate target network is used to compute stable Q-value targets. It is periodically synced with the main network to avoid oscillating targets.
  
- Epsilon-Greedy Exploration: The agent starts by exploring random actions (high epsilon) and gradually shifts towards exploitation (lower epsilon) as it learns more about the environment.
  
- Reward Function: Designed to capture both terminal outcomes (win/loss) and intermediate progress (damage dealt, Pokémon fainted, status applied), encouraging consistent and strategic play.
  
- Double DQN: To reduce overestimation bias, the agent uses the policy network to select the best next action, and the target network to evaluate it.

---

## Results

We tested the DQN Agent against a set of 3 standard players provided by the *poke-env* library:

- **RandomPlayer**: Selects a random action at each turn
- **MaxBasePowerPlayer**: A greedy player, selects the move with the highest base power at each turn.
- **SimpleHeuristicsPlayer**: A human-like player which understands effective switching and type matchups.

Since we had 6 Pokémon team choices for both the player and opponent, and 3 opponents, we had a total of 108 possible configurations. The DQN agent was trained on each of these configurations for 5000 episodes, and then evaluated for 100 episodes. The evaluation results are given below.

### Win rates against RandomPlayer

| Opponent Team \ Agent Team | Team1 | Team2 | Team3 | Team4 | Team5 | Team6 |
|----------------------------|-------|-------|-------|-------|-------|-------|
| Team1                     |   0   |   0   |   0   |   0   |   0   |   0   |
| Team2                     |   0   |   0   |   0   |   0   |   0   |   0   |
| Team3                     |   0   |   0   |   0   |   0   |   0   |   0   |
| Team4                     |   0   |   0   |   0   |   0   |   0   |   0   |
| Team5                     |   0   |   0   |   0   |   0   |   0   |   0   |
| Team6                     |   0   |   0   |   0   |   0   |   0   |   0   |

### Win rates against MaxBasePowerPlayer

| Opponent Team \ Agent Team | Team1 | Team2 | Team3 | Team4 | Team5 | Team6 |
|----------------------------|-------|-------|-------|-------|-------|-------|
| Team1                     |   0   |   0   |   0   |   0   |   0   |   0   |
| Team2                     |   0   |   0   |   0   |   0   |   0   |   0   |
| Team3                     |   0   |   0   |   0   |   0   |   0   |   0   |
| Team4                     |   0   |   0   |   0   |   0   |   0   |   0   |
| Team5                     |   0   |   0   |   0   |   0   |   0   |   0   |
| Team6                     |   0   |   0   |   0   |   0   |   0   |   0   |

### Win rates against SimpleHeuristicsPlayer

| Opponent Team \ Agent Team | Team1 | Team2 | Team3 | Team4 | Team5 | Team6 |
|----------------------------|-------|-------|-------|-------|-------|-------|
| Team1                     |   0   |   0   |   0   |   0   |   0   |   0   |
| Team2                     |   0   |   0   |   0   |   0   |   0   |   0   |
| Team3                     |   0   |   0   |   0   |   0   |   0   |   0   |
| Team4                     |   0   |   0   |   0   |   0   |   0   |   0   |
| Team5                     |   0   |   0   |   0   |   0   |   0   |   0   |
| Team6                     |   0   |   0   |   0   |   0   |   0   |   0   |

---

## Observations

---

## Challenges

The biggest challenge was designing a good state embedding which effectively captured all information in a single battle state. Initially we kept a very simple and sparse state vector with very limited information, and the results were not good. The partially observable nature of the game also made it harder to encode opponent information into the state vector. Comparatively, designing the reward function was much easier, and even a simple reward function gave good results.

Designing the teams was another challenge. We have 4 balanced teams along with one slightly stronger and one slightly weaker team, to check if the agent is winning just because of a good team, or if it has actually learnt the optimal moves at each step. The results show that the agent is indeed winning more with a stronger team, and less with a weaker team, but it is a slight difference, and the agent continues to win the majority of the battles against the random and greedy players.

The last challenge was hyperparameter tuning. There are a lot of tunable parameters, and training one configuration of agent team, opponent team, and agent class takes quite a bit of time. So finding the right parameters to have the maximum winrate was a time-consuming process.

---

## Steps to Run

Python >= 3.10 is required. To install the other requirements, run

```none
pip install -r requirements.txt
```

In the `Pokémon-showdown` directory, run

```none
npm install
node Pokémon-showdown start --no-security
```

This will start a Showdown server at  `localhost:8000`. This needs to be running for training and evaluation.

*Note: Server may not run on IIITB-Milan, a hotspot may be required.*

The DQNAgent class is implemented in `src/agent.py`.

To train the model, navigate to `src` and run

```none
python train.py
```

This will start a training loop with the configuration specified in `train.py`. After training, the model will be saved in `models`.

To evaluate the model, navigate to `src` and run

```none
python evaluate.py
```

This will try to find a model with the configuration specified in `evaluate.py`, and if it exists, evaluate its performance.

---

## Contributions

[IMT2022076 Mohit Naik](https://github.com/mohit065): Implemented the `agent.py` file, and designed the teams in `teams.py`.

[IMT2022086 Ananthakrishna K](https://github.com/Ananthakrishna-K-13): Implemented the `train.py` and `evaluate.py` files along with hyperparameter tuning for the DQN agent.

---
