# Chef’s Hat Gym - Reinforcement Learning Project

---

##  Source Code

This repository contains the full source code for a Reinforcement Learning agent trained on the Chef’s Hat Gym environment.

### Included Components:
- PPO-based Agent (Actor-Critic architecture)
- Training script (`train.py`)
- Environment interaction code (`src/`)
- Opponent modelling extension
- Utility functions for state processing and action mapping

---

##  Assignment Variant

**Student ID Modulo 7 = 1**
#### Variant: Opponent Modelling Extension

This project extends a PPO-based reinforcement learning agent by incorporating opponent behaviour modelling. The agent learns not only from environment rewards but also by interacting with stochastic opponent strategies, improving adaptability in a multi-agent environment.

---

##  How to Install Dependencies

Install all required packages using:
pip install -r requirements.txt

#### Main Dependencies:
- TensorFlow / Keras
- NumPy
- Gym-style environment (Chef’s Hat Gym)

## How to Run the Code

To train the reinforcement learning agent:

python train.py

####  The training executes as:
- The game environment is initialized
- The agent plays multiple rounds of the game
- It selects actions using a neural network
- It learns from rewards based on performance
- Its decision-making gradually improves over time

 ## Experimental Outputs

During training, the following outputs are generated:

### Model Outputs:
1. agent_actor.h5 → Helps the agent decide which action to take (Actor)
2. agent_critic.h5 → Helps evaluate game situations (Critic)

### Experiments Conducted
The following experiments were performed:

1. PPO Agent Training
Baseline reinforcement learning agent trained using PPO algorithm
2. Opponent Modelling
Agents trained against stochastic/random opponents
Analysis of non-stationary environment effects
3. Action Mapping Strategy
Converted environment-specific actions into discrete agent decisions
4. State Representation Testing
Flattened observation vectors used for neural network input

## Results
- Agent improves gradually over episodes
- Performance depends on opponent randomness
- Stable learning using PPO clipped objective
  
- Increasing rewards over episodes indicate learning progress
- Actor model improves action selection policy
- Critic model estimates state value accuracy
- Performance variation occurs due to opponent randomness

## Limitations
- Training time was limited, so the model was not fully optimized
- Opponent strategies are relatively simple (random-based)
- More training episodes could improve performance further
- Reward tuning could be improved for better learning efficiency

## Conclusion

This project demonstrates how reinforcement learning can be used to train an AI agent to play a complex card game.

Instead of following fixed rules, the agent learns through trial and error by interacting with the environment.

The opponent modelling variant makes the task more realistic, as the agent have to adapt to different types of opponents rather than playing against a fixed strategy.

Overall, the project successfully demonstrates the core ideas of reinforcement learning, PPO, and adaptive decision-making in a game environment.
