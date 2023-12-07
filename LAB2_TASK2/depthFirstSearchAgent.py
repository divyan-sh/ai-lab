from poker_game_example import PokerPlayer
import poker_game

class DepthFirstAgent(PokerPlayer):
    def __init__(self, stack=400):
        super().__init__(current_hand_=None, stack_=stack, action_=None, action_value_=None)
        self.amountOfNodesExtended = 0

    def evaluateState(self, stateQueue):
        if stateQueue:
            currentState = stateQueue.pop()
            nextStates = poker_game.generate_successor_states(currentState)
            self.amountOfNodesExtended += len(nextStates)
            return nextStates
        return []
