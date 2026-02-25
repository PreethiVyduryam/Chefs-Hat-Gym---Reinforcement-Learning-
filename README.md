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
