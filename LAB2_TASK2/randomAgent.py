import random
import poker_environment as pe
from poker_game_example import PokerPlayer
import poker_game

class RandomAgent(PokerPlayer):
    def __init__(self, current_hand=None, stack=300, action=None, action_value=None):
       super().__init__(current_hand_=current_hand, stack_=stack, action_=action, action_value_=action_value)
        # Rest of the RandomAgent code

    def evaluateState(self, stateQueue):
        if len(stateQueue) == 0:
            return []
        randIndex = random.randint(0, len(stateQueue) - 1)
        selectedState = stateQueue.pop(randIndex)
        nextStates = poker_game.generate_successor_states(selectedState)
        return nextStates
