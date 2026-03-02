import os
import sys
import numpy as np
from agentPPO import AgentPPO
from opponent_model import OpponentModelWrapper

# Ensure the ChefsHat environment is in path
CHEFS_HAT_PATH = os.path.join(os.getcwd(), "src")
sys.path.append(CHEFS_HAT_PATH)

from game_env.game import Game  # original import

# Initialize environment
player_names = ["Alice", "Bob", "Charlie", "David"]
game = Game(player_names)

# Initialize agent
state_size = 20  # Example: depends on observation representation
action_size = 10  # Example: number of possible actions
agent = AgentPPO(state_size, action_size)

# Initialize opponent model
opponent_model = OpponentModelWrapper(player_names)

# Training loop placeholder
episodes = 100
for ep in range(episodes):
    game.create_new_match()
    game.start_match()
    done = False
    while not done:
        obs = game.step()  # get observation
        action, prob = agent.get_action(obs)
        result = game.step(action)
        done = result.get("match_over", False)

# Save models
agent.actor.save("agent_actor.h5")
agent.critic.save("agent_critic.h5")
