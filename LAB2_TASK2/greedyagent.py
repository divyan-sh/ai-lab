import numpy as np
from poker_game_example import PokerPlayer
import poker_game
import poker_environment as environment

class GreedyAgent(PokerPlayer):
    def __init__(self, current_hand=None, stack=400, action=None, action_value=None):
        super().__init__(current_hand_=current_hand, stack_=stack, action_=action, action_value_=action_value)
        self.amountOfNodesExtended = 0
        self.hasDepthLimit = False

    def evaluate_hand_strength(self):
        # Simple hand strength evaluation based on the hand type
        return environment.Types[self.current_hand_type[0]] * 13 + environment.Ranks[self.current_hand_type[1]]

    def evaluateState(self, stateQueue):
        best_state = None
        best_gain = -np.inf

        for state in stateQueue:
            hand_strength = self.evaluate_hand_strength()
            potential_gain = state.agent.stack - self.stack

            if hand_strength >= 10:  # Adjust threshold based on hand strength
                if potential_gain > best_gain:
                    best_gain = potential_gain
                    best_state = state

        next_states = poker_game.generate_successor_states(best_state) if best_state else []
        return next_states
