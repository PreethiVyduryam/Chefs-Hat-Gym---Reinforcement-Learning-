# Chefs-Hat-Gym---Reinforcement-Learning-
## Assigned Variant

Student ID modulo 7 = 1
Variant: Opponent Modelling Extension

This project extends a PPO-based reinforcement learning agent in the Chef’s Hat Gym environment by incorporating opponent behaviour modelling. The agent learns not only from environment rewards but also from predicted opponent strategies.

## Environment

Environment: Chef’s Hat Gym
Framework: TensorFlow + PPO

The agent interacts with the Chef’s Hat card game environment and learns optimal policies through policy gradient optimisation.

## How to Install Dependencies
pip install -r requirements.txt

## How to Train the Agent
python train.py

Which will:

* Train the PPO agent

* Train the opponent model

* Save trained model weights

* Generate reward and win-rate plots in the results folder

## How to Evaluate the Agent
python evaluate.py

Which will:

* Load saved models

* Run evaluation episodes

* Print win rate and average reward

* Save evaluation results in results folder

## Experiments Conducted

The following experiments were performed during this projwct:

* Baseline PPO agent without opponent modelling.

* PPO agent with opponent modelling extension.


## Results Interpretation

* The opponent modelling agent achieved higher win rate.

* Faster convergence observed in reward curve.

* Demonstrates that modelling opponent behaviour improves strategic adaptation.

* Population-based training
* Self-play dynamics
* Multi-agent learning


The opponent modelling extension was evaluated against a baseline PPO agent. Performance was measured using average reward and win rate. Results indicate whether modelling opponent behaviour improves strategic performance in the Chef’s Hat environment.

- reset() → returns initial state  
- step(actions) → returns (next_state, reward, done, info)  
- fixed random seeds for reproducibility  
- valid‑actions logic  
- turn‑based multi‑agent behaviour  

### State Representation
The agent receives a **10‑dimensional continuous vector** representing abstracted game information such as:
- simplified current hand structure  
- turn-related features  
- episode progress  
Since Chef’s Hat is partially observable and high‑dimensional. A compressed numerical state:
- reduces complexity  
- stabilises PPO training  
- follows typical RL design in card-game research  
### Action Handling Strategy
- Chef’s Hat is a discrete decision-making game  
- action masking improves sample efficiency  
- ensures realistic legality of moves  
- commonly used in card‑game RL (Poker, Uno, Hearthstone)
### Reward Usage
Rewards used:
- small step rewards (±0.1) to maintain learning signal  
- terminal win/loss reward (+1 / −1)
  
Chef’s Hat has long horizons, sparse-only rewards make learning unstable.  
Reward shaping improves convergence and strategy discovery.

### RL Algorithm Selection (PPO Justification)
The project uses **Proximal Policy Optimisation (PPO)**.
- PPO is stable for multi-agent environments  
- widely used in games (Dota2, Hearthstone, MuZero variants)  
- supports policy clipping to prevent destabilising updates  
- performs well with continuous states + discrete actions  
- integrates cleanly with TensorFlow  
This makes PPO a well‑justified and academically strong choice.

### Evaluation Metrics
The agent is evaluated using:
- **average episode reward**  
- **win rate approximation**  
- **reward convergence curves**  
- **performance across different opponent types**  
- **stability over multiple episodes**
Plots and metrics will in case be saved to the 'results/' folder.

### Limitations, Challenges, and Failure Modes
- Simplified environment does not include full Chef’s Hat rules  
- Multi-agent non-stationarity can destabilize policies  
- PPO is sensitive to learning rate and clipping hyperparameters  
- Some opponent heuristics are simplified  
- Reward shaping may bias behaviour  
- Lack of full observability makes long-term planning difficult  
These limitations are common in multi-agent RL and are acknowledged in the analysis.
