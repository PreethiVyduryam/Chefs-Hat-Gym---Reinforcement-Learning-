# Chefs-Hat-Gym---Reinforcement-Learning-
This project explores reinforcement learning in a competitive multi-agent environment using the Chef’s Hat Gym framework. The objective is to design, implement, train, and critically analyse reinforcement learning agents capable of playing the Chef’s Hat card game effectively.

This submission corresponds to the Opponent Modelling Variant (Student ID mod 7 = 1), focusing on:
  
  * The impact of different opponent behaviours
  
  * Training against various baselines
  
  * Analysis of non-stationarity in multi-agent learning
  
  * Performance comparison under different opponent modelling
The project emphasises both technical implementation and critical evaluation.

## Environment:

The official Chef’s Hat Gym environment was used:

GitHub: https://github.com/pablovin/ChefsHatGYM

Documentation: https://chefshatgym.readthedocs.io/en/latest/

## Key characteristics of the environment:

  * Multi-agent competitive interaction

  * Turn-based gameplay

* Large discrete action space

* Delayed and sparse rewards

* Stochastic and non-stationary dynamics

## State Representation:

The agent’s state representation includes:

* Current hand information

* Game phase indicators

* Previous actions

* Public game state information

The representation preserves relevant strategic information while maintaining computational feasibility.

**Justification** 
The chosen representation allows the agent to reason about both immediate legal moves and longer-term strategy, while adapting to opponent behaviour.

## Action Handling Strategy

Chef’s Hat has a large and variable discrete action space.

Action handling includes:

* Encoding valid actions only

* Masking illegal actions

* Mapping action indices to environment-compatible actions

* This ensures stable learning and avoids invalid move penalties.

## Reward Structure

The default environment reward was used:

* Sparse and delayed reward assigned at end of match

* Positive reward for winning

* Negative reward for losing

**Justification:**
Using the environment’s natural reward structure allows analysis of how the agent learns under delayed credit assignment and competitive pressure.

## Reinforcement Learning Algorithm

The primary algorithm implemented is:

* Proximal Policy Optimization (PPO)

**Justification:**

* Stable policy-gradient method

* Performs well in large discrete action spaces

* Robust to noisy and non-stationary environments

* Suitable for multi-agent competitive settings

**Hyperparameters include:**

* Learning rate

* Discount factor (gamma)

* Clipping parameter

* Batch size

* Entropy coefficient

Hyperparameters were tuned experimentally.

## Training Procedure

Training was conducted through iterative interaction with the environment:

* Agent interacts with Chef’s Hat Gym.

* Experiences are collected.

* Policy updates are performed using PPO.

* Performance is evaluated periodically.

Training was performed over multiple episodes to allow convergence.

## Opponent Modelling Experiments

As required by the Opponent Modelling Variant, experiments included:

1. Training Against Fixed Baseline Opponent

* Random agent baseline

* Rule-based heuristic opponent

2. Training Against Mixed Opponent Strategies

* Alternating opponent types

* Increasing opponent difficulty

3. Evaluation Against Unseen Opponent Configuration

* Generalisation testing

* Stability assessment

## Evaluation Metrics

Agent performance was evaluated using:

* Win rate over multiple matches

* Environment performance score

* Learning curves (reward vs training steps)

* Policy stability across seeds

Results were aggregated over multiple evaluation runs.

## Experimental Results

Key findings:

* Agent performance improved significantly when trained against varied opponent strategies.

* Training against a single opponent led to overfitting behaviour.

* Non-stationarity influenced convergence speed.

* Mixed opponent training improved robustness and generalisation.

Learning curves demonstrate gradual improvement in win rate and stabilisation of policy performance.

## Limitations

* Multi-agent non-stationarity increases variance in learning.

* Sparse rewards slow early training.

* Large action space increases exploration complexity.

* Training time is computationally intensive.

## Reproducibility

To reproduce results:

1. Clone this repository.

2. Install required dependencies:

pip install -r requirements.txt

3. Run training:

python train.py

4. Evaluate:

python evaluate.py

Random seeds are fixed where applicable for reproducibility.

## Conclusion

This project demonstrates that reinforcement learning agents in competitive multi-agent environments must account for opponent behaviour and non-stationarity. Training against diverse opponent strategies improves robustness and generalisation. PPO proved to be a stable and effective method for the Chef’s Hat Gym environment.

**Future work may explore:**

* Explicit opponent modelling networks

* Population-based training

* Self-play dynamics

* Multi-agent learning frameworks
