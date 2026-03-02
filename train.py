# train.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tqdm import trange

try:
    from core.game_env.game import Game
    from core.opponent_model import OpponentModelWrapper
except ImportError:
    # fallback for some setups
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "src/core"))
    from game_env.game import Game
    from opponent_model import OpponentModelWrapper

# PPO Hyperparameters
GAMMA = 0.99
CLIP_EPSILON = 0.2
ACTOR_LR = 1e-4
CRITIC_LR = 5e-4
BATCH_SIZE = 32
EPOCHS = 3

# Define Actor and Critic Networks
def build_actor(input_shape, action_dim):
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(action_dim, activation='softmax')(x)
    return Model(inputs, outputs)

def build_critic(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(1, activation='linear')(x)
    return Model(inputs, outputs)

# PPO Agent
class AgentPPO:
    def __init__(self, obs_dim, action_dim):
        self.actor = build_actor(obs_dim, action_dim)
        self.critic = build_critic(obs_dim)
        self.actor_optimizer = Adam(learning_rate=ACTOR_LR)
        self.critic_optimizer = Adam(learning_rate=CRITIC_LR)
        self.gamma = GAMMA
        self.clip_epsilon = CLIP_EPSILON

    def get_action(self, state):
        state = np.expand_dims(state, axis=0).astype(np.float32)
        probs = self.actor(state).numpy()[0]
        action = np.random.choice(len(probs), p=probs)
        return action, probs[action]

    def compute_returns(self, rewards, dones, last_value):
        returns = []
        R = last_value
        for r, done in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - done)
            returns.insert(0, R)
        return np.array(returns)

    def update(self, states, actions, old_probs, returns):
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions)
        old_probs = np.array(old_probs)
        returns = np.array(returns, dtype=np.float32)

        for _ in range(EPOCHS):
            with tf.GradientTape(persistent=True) as tape:
                # Critic loss
                values = tf.squeeze(self.critic(states), axis=1)
                critic_loss = tf.reduce_mean(tf.square(returns - values))

                # Actor loss
                probs = self.actor(states)
                action_masks = tf.one_hot(actions, probs.shape[1])
                selected_probs = tf.reduce_sum(action_masks * probs, axis=1)
                ratios = selected_probs / (old_probs + 1e-10)
                adv = returns - values
                actor_loss = -tf.reduce_mean(tf.minimum(ratios * adv, tf.clip_by_value(ratios, 1-self.clip_epsilon, 1+self.clip_epsilon) * adv))

            actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
            critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
            self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        del tape

# Flatten game observation into numerical vector
def flatten_observation(obs):
    hand = obs.get('hand', [])
    board = obs.get('board', [])
    flat = np.zeros(28)
    flat[:len(hand)] = hand
    flat[14:14+len(board)] = board
    return flat

# Main Training Loop
if __name__ == "__main__":
    # Create the game environment
    players = ["Agent", "Opponent1", "Opponent2", "Opponent3"]
    game_env = Game(players, max_matches=5)
    opponent_model = OpponentModelWrapper(players[1:])

    obs_dim = 28  # 14 for hand + 14 for board
    action_dim = 17  # maximum possible cards to play

    agent = AgentPPO(obs_dim, action_dim)

    EPISODES = 50
    for ep in trange(EPISODES, desc="Training Episodes"):
        game_env.create_new_match()
        game_env.start_match()
        done = False

        # Experience buffers
        states, actions, rewards, dones, old_probs = [], [], [], [], []

        while not done:
            step_info = game_env.step()
            if "request_action" in step_info:
                player_name = step_info["player"]
                obs = step_info["observation"]
                state_vec = flatten_observation(obs)

                # If opponent turn, use opponent model
                if player_name != "Agent":
                    action = opponent_model.predict_action(player_name, obs)
                    prob = 1.0
                else:
                    action, prob = agent.get_action(state_vec)
                    states.append(state_vec)
                    actions.append(action)
                    old_probs.append(prob)

                # Execute action
                step_info = game_env.step(action)
                reward = step_info.get("reward", 0)
                done = step_info.get("match_over", False)

                rewards.append(reward)
                dones.append(done)

            else:
                # If no action requested, continue
                done = step_info.get("match_over", False)

        # Compute returns and update PPO agent
        if len(states) > 0:
            last_value = 0  # assuming 0 at end of match
            returns = agent.compute_returns(rewards, dones, last_value)
            agent.update(states, actions, old_probs, returns)

    # Save trained weights
    agent.actor.save("agent_actor.h5")
    agent.critic.save("agent_critic.h5")
    print("Training complete! Actor and Critic models saved.")
