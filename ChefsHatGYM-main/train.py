# ================================
# FIXED PATH SETUP (CRITICAL)
# ================================
import sys
import os

# Add src folder to Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

sys.path.insert(0, SRC_PATH)

print("Using SRC path:", SRC_PATH)

# ================================
# IMPORTS (NOW WILL WORK)
# ================================
from core.game_env.game import Game
class DummyLogger:
    def engine_log(self, msg):
        pass
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# ================================
# SIMPLE PPO AGENT
# ================================
class AgentPPO:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.actor = self.build_actor()
        self.critic = self.build_critic()

    def build_actor(self):
        model = tf.keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(self.state_size,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.action_size, activation='softmax')
        ])
        return model

    def build_critic(self):
        model = tf.keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(self.state_size,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(1)
        ])
        return model

    def get_action(self, state):
        state = np.expand_dims(state, axis=0)
        probs = self.actor(state).numpy()[0]
        action = np.random.choice(len(probs), p=probs)
        return action

# ================================
# HELPER
# ================================
def flatten_obs(obs):
    hand = obs.get("hand", [])
    board = obs.get("board", [])

    vec = np.zeros(28)
    vec[:len(hand)] = hand
    vec[14:14+len(board)] = board
    return vec

# ================================
# TRAINING LOOP
# ================================
# ================================
# TRAINING LOOP
# ================================

players = ["Agent", "P1", "P2", "P3"]
game = Game(players, logger=DummyLogger())

agent = AgentPPO(state_size=28, action_size=17)

EPISODES = 1
MAX_STEPS = 50

for ep in range(EPISODES):

    print(f"Starting Episode {ep+1}")

    step_count = 0

    game.create_new_match()
    game.start_match()

    done = False

    while not done:

        step_count += 1
        if step_count > MAX_STEPS:
            print("Max steps reached, ending episode")
            break

        step = game.step()

        if "request_action" in step:

            player = step["player"]
            obs = step["observation"]
            possible_actions = obs.get("possible_actions", [])

            if player == "Agent":

                state = flatten_obs(obs)

                action_idx = agent.get_action(state)

                # safety clamp
                if len(possible_actions) == 0:
                    action = "pass"
                else:
                    action_idx = action_idx % len(possible_actions)
                    action = possible_actions[action_idx]

            else:
                import random
git push
                if possible_actions:
                    action = random.choice(possible_actions)
                else:
                    action = "pass"
            step = game.step(action)

        done = step.get("match_over", False)

# ================================
# SAVE MODELS
# ================================
agent.actor.save("agent_actor.h5")
agent.critic.save("agent_critic.h5")

print("\nSUCCESS: Models saved!")
