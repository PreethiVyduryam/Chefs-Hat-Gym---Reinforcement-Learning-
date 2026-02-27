import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from opponent_model import OpponentModelWrapper, BaseOpponent
import random


def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

set_seed(42)


class PPOAgent:
    def __init__(self, state_size, action_size, lr=0.0003, gamma=0.99, clip=0.2):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.clip = clip

        self.actor = self.build_actor(lr)
        self.critic = self.build_critic(lr)

    def build_actor(self, lr):
        inp = layers.Input(shape=(self.state_size,))
        x = layers.Dense(128, activation="relu")(inp)
        x = layers.Dense(128, activation="relu")(x)
        out = layers.Dense(self.action_size, activation="softmax")(x)

        model = tf.keras.Model(inp, out)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
        return model

    def build_critic(self, lr):
        inp = layers.Input(shape=(self.state_size,))
        x = layers.Dense(128, activation="relu")(inp)
        x = layers.Dense(128, activation="relu")(x)
        out = layers.Dense(1)(x)

        model = tf.keras.Model(inp, out)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="mse"
        )
        return model

    def select_action(self, state, valid_actions):
        probs = self.actor.predict(state[np.newaxis], verbose=0)[0]

        mask = np.zeros_like(probs)
        mask[valid_actions] = 1
        probs = probs * mask

        # prevent divide-by-zero
        if np.sum(probs) == 0:
            probs = mask / np.sum(mask)
        else:
            probs = probs / np.sum(probs)

        action = np.random.choice(len(probs), p=probs)
        return action, probs[action]

    def train_step(self, states, actions, old_probs, rewards, values):

        # Compute discounted returns
        discounted = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            discounted.insert(0, G)

        discounted = np.array(discounted)
        advantages = discounted - values

        with tf.GradientTape() as tape:
            new_probs = self.actor(states, training=True)
            action_masks = tf.one_hot(actions, self.action_size)
            new_probs = tf.reduce_sum(action_masks * new_probs, axis=1)

            ratio = new_probs / (old_probs + 1e-10)
            clipped = tf.clip_by_value(ratio, 1 - self.clip, 1 + self.clip)

            loss = -tf.reduce_mean(
                tf.minimum(ratio * advantages,
                           clipped * advantages)
            )

        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(
            zip(grads, self.actor.trainable_variables)
        )

        self.critic.train_on_batch(states, discounted)


def train(episodes=2000):

    env = OpponentModelWrapper(
        gym.make("chefshat-v0"),
        opponent_class=BaseOpponent
    )

    state_size = env.env.observation_space.shape[0]
    action_size = env.env.action_space.n

    agent = PPOAgent(state_size, action_size)

    win_history = []

    for ep in range(episodes):

        state = env.reset()
        done = False

        states, actions, old_probs, rewards, values = [], [], [], [], []

        while not done:
            valid = env.env.get_valid_actions(0)

            action, prob = agent.select_action(state, valid)
            value = agent.critic.predict(state[np.newaxis], verbose=0)[0][0]

            next_state, reward, done, info = env.step(action)

            states.append(state)
            actions.append(action)
            old_probs.append(prob)
            rewards.append(reward)
            values.append(value)

            state = next_state

        # Determine win/loss from final reward
        episode_reward = sum(rewards)

        if episode_reward > 0:
            win_history.append(1)
        else:
            win_history.append(0)

        agent.train_step(
            np.array(states),
            np.array(actions),
            np.array(old_probs),
            np.array(rewards),
            np.array(values)
        )

        # Print progress every 50 episodes
        if ep % 50 == 0 and ep > 0:
            win_rate = np.mean(win_history[-50:])
            print(f"Episode {ep} | Reward: {episode_reward:.2f} | WinRate(last50): {win_rate:.2f}")

    # Save models
    agent.actor.save("ppo_actor.h5")
    agent.critic.save("ppo_critic.h5")

    # Save win history
    np.save("win_history.npy", np.array(win_history))

    print("Training complete.")


if __name__ == "__main__":
    train()
