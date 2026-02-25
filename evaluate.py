import gym
import numpy as np
import tensorflow as tf
from opponent_model import OpponentModelWrapper, AggressiveOpponent

def evaluate(episodes=50):
    env = OpponentModelWrapper(
        gym.make("chefshat-v0"),
        opponent_class=AggressiveOpponent
    )

    actor = tf.keras.models.load_model("ppo_actor.h5")

    wins = 0

    for ep in range(episodes):
        state = env.reset()
        done = False
        while not done:
            valid = env.env.get_valid_actions(0)
            probs = actor.predict(state[np.newaxis], verbose=0)[0]

            mask = np.zeros_like(probs)
            mask[valid] = 1
            probs = probs * mask
            probs = probs / np.sum(probs)

            action = np.random.choice(len(probs), p=probs)
            state, reward, done, info = env.step(action)

        if reward == 1:  # or check win condition from info
            wins += 1

    print("Win rate:", wins / episodes)

if __name__ == "__main__":
    evaluate()
