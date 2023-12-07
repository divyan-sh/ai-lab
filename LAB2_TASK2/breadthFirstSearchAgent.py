import poker_environment as environment
from poker_game_example import PokerPlayer
import poker_game

class BreadthFirstAgent(PokerPlayer):
    def __init__(self, current_hand=None, stack=400, action=None, action_value=None):
        super().__init__(current_hand_=current_hand, stack_=stack, action_=action, action_value_=action_value)
        self.amountOfNodesExtended = 0
        self.hasDepthLimit = False

    def evaluateState(self, stateQueue):
        if stateQueue:
            currentState = stateQueue.pop(0)
            nextStates = poker_game.generate_successor_states(currentState)
            self.amountOfNodesExtended += len(nextStates)
            return nextStates
        return []
