import numpy as np

class BaseOpponent:
    #"""Default baseline opponent – random valid action."""
    def select_action(self, valid_actions):
        return np.random.choice(valid_actions)


class AggressiveOpponent(BaseOpponent):
    #"""Prefers high-value actions (example heuristic)."""
    def select_action(self, valid_actions):
        return max(valid_actions)


class PassiveOpponent(BaseOpponent):
    #"""Prefers low-value actions."""
    def select_action(self, valid_actions):
        return min(valid_actions)


class OpponentModelWrapper:
    #"""
    #Wraps the environment and controls opponents' actions.
    #Assumes env.step() receives joint actions [player, opp1, opp2, opp3].
    #"""
    def __init__(self, env, opponent_class=BaseOpponent):
        self.env = env
        self.opponents = [opponent_class(), opponent_class(), opponent_class()]

    def reset(self):
        return self.env.reset()

    def step(self, player_action):
        opponent_actions = []
        for i, opp in enumerate(self.opponents):
            valid = self.env.get_valid_actions(player_id=i+1)
            opponent_actions.append(opp.select_action(valid))

        actions = [player_action] + opponent_actions
        return self.env.step(actions)

    def render(self):
        self.env.render()
