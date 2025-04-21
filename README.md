# RLmon - A Pokemon Showdown Bot Using Reinforcement Learning

## Index
- [Overview](#overview)
- [Setting the Game](#setting-the-game)
- [Setting the MDP](#setting-the-mdp)
- [Q-Learning](#q-learning)
- [Results](#results)
- [Observations](#observations)
- [Conclusion](#conclusion)
- [Steps to Run](#steps-to-run)
- [Contributions](#contributions)

---

## Overview

This project aims to implement, train and evaluate a reinforcement learning agent to play Pokemon battles on the Pokemon Showdown battle simulator. We first formulate a Pokemon battle as a *Markov Decision Process* (MDP), then train the agent to learn the optimal policy at each state via Deep Q-Learning, and evaluate its performance against a set of standard opponents.

---

## Setting the Game

A Pokemon battle between a player and an opponent is a turn-based game where both sides have 6 pokemon each. At any point of time, exactly one pokemon from each side is *active* i.e. out for battle. At each turn, both the player and the opponent can either use their active pokemon to make a move, or switch out their active pokemon. The objective of the game is to defeat all 6 of the opponent's pokemon first.

Each pokemon has a number of statistics associated with it:
- Type: Each pokemon can have one or two types.
- Status Conditions: During the course of the battle, a pokemon can get afflicted with one of several possible status conditions (sleep, paralysis, burn, frost, poison), each of which has different effects on the other statistics of the pokemon.
- Moves: Each pokemon has a set of 4 moves, each of which have their own statistics:
  - Base Power: The damaging power of a move. Moves with higher base power will cause higher damage to the opponent pokemon.
  - Accuracy: The chance of a move hitting the opponent.
  - Move category: Moves are categorized as physical, special and status.
  - Priority: Some moves with higher priority always go first.
  - Type: Each move has exactly one type, chosen from the set of pokemon types. Moves with types matching the pokemon playing that move get an increase in base power. Moves of a certain type may have higher effect / lower effect / no effect on pokemon of a certain type, which is computed using a type matchup table.
  - PP: The number of times you can use a single move
  - Secondary effect: Some moves may have a secondary effect like increasing the stats of the active pokemon or inflicting a status condition on the opponent.
- HP: The hit-points of a pokemon. A damaging move will reduce the hit-points, and the pokemon is defeated when the HP falls to zero.
- Atk / SpAtk: The attacking power of a pokemon, which influences the base power of its moves.
- Def / SpDef: The defensive power of a pokemon which influences how much HP it loses when hit by moves.
- Speed: A pokemon's speed. Moves of faster pokemon will go first.
- Abilities, Items, Natures: Each pokemon has an ability and a nature, and can hold an item. These may boost one of its stats, reduce those of the opponent etc.

Note: To reduce complexity and focus on the core problem, we have ignored some functionalities like weather conditions, secondary status conditions like confusion/infatuation/flinching, single type pokemon, complex moves, items and abilities, IVs. Also, the battle format used is gen4anythinggoes, which automatically discards Mega Evolutions, Z-Moves, Dynamax/Galarian Pokemon, and Terrastalize. This ensures consistency and simplicity while keeping the core philosophy of a pokemon battle intact.

We created 6 balanced teams of 6 dual-type pokemon each, with every pokemon having an item (Leftovers), an ability, a nature, EVs to boost stats, and a set of 4 moves.

---

## Setting the MDP

---

## Q-Learning

---

## Results

---

## Observations

---

## Conclusion

---

## Steps to Run

---

## Contributions

---

Github Link : https://github.com/mohit065/rlmon
