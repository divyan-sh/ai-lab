import poker_environment as env
from poker_environment import AGENT_ACTIONS, BETTING_ACTIONS
from poker_game_example import PokerPlayer, GameState
import copy

MAX_HANDS = 4
INIT_AGENT_STACK = 400
MAX_EXPANSIONS = 10000

class PokerGameSimulation:

    def __init__(self, player_agent):
        self.player = player_agent
        self.rival = PokerPlayer(None, INIT_AGENT_STACK)
        self.initial_state = GameState(0, 0, 'INIT_DEALING', 0, None, player_agent, self.rival)
        self.game_states = []
        self.is_game_active = True
        self.final_state = None
        self.dealt_hands = 0
        self.expanded_nodes = 0

    def check_game_termination(self, state):
        if state.phase == 'SHOWDOWN' and (state.player.stack >= 500 or self.player.node_limit_reached(MAX_EXPANSIONS)):
            return True
        return False

    def play_game(self):
        hands_played = 0

        while self.is_game_active:
            if self.game_states:
                next_states = self.player.evaluateState(self.game_states.pop())
            else:
                next_states = generate_successor_states(self.initial_state)

            self.game_states.extend(next_states)
            self.expanded_nodes += len(next_states)

            for state in next_states:
                if self.check_game_termination(state):
                    self.final_state = state
                    self.is_game_active = False
                    hands_played = state.nn_current_hand  # Assuming nn_current_hand tracks the number of hands played

        # Assuming the final stack is stored in self.final_state.agent.stack
        final_stack = self.final_state.agent.stack if self.final_state else None
        return final_stack, hands_played

    def display_game_results(self):
        current = self.final_state
        path_length = 0

        while current.parent_state:
            current = current.parent_state
            path_length += 1

        game_result = {
            "Result": "Win" if self.final_state.player.stack > self.final_state.rival.stack else "Loss",
            "Player_Stack": self.final_state.player.stack,
            "Rival_Stack": self.final_state.rival.stack,
            "Path_Length": path_length,
            "Nodes_Expanded": self.expanded_nodes,
            "Total_Hands_Dealt": self.final_state.nn_current_hand
        }

        print(game_result)

    
def generate_successor_states(current_state):
    if not current_state or not current_state.agent:
        # Handle error or incorrect state
        print("Error: Current state or agent is None.")
        return []

    next_states = []

    if current_state.phase in ['SHOWDOWN', 'INIT_DEALING'] or current_state.acting_agent == 'rival':
        for action in current_state.agent.get_actions():
            state_clone = copy_state(current_state)
            state_clone.acting_agent = 'player'

            if current_state.phase != 'BIDDING':
                state_clone.deal_cards()

            process_action(state_clone, action)
            next_states.append(state_clone)

    elif current_state.phase == 'BIDDING' and current_state.acting_agent == 'player':
        rival_action, value = env.poker_strategy_example(current_state.opponent.current_hand_type[0],
                                                         current_state.opponent.current_hand_type[1],
                                                         current_state.opponent.stack,
                                                         current_state.agent.action,
                                                         current_state.agent.action_value,
                                                         current_state.agent.stack,
                                                         current_state.pot,
                                                         current_state.nn_current_bidding)
        process_rival_action(current_state, rival_action, value, next_states)

    return next_states

def process_action(state, action):
    if action == 'CALL':
        if state.player.stack >= 5:
            state.phase = 'SHOWDOWN'
            state.player.action = action
            state.player.action_value = 5
            state.player.stack -= 5
            state.pot += 5

            state.showdown()

            state.nn_current_hand += 1
            state.nn_current_bidding = 0
            state.pot = 0

    elif action == 'FOLD':
        state.phase = 'SHOWDOWN'
        state.player.action = action
        state.rival.stack += state.pot

        state.nn_current_hand += 1
        state.nn_current_bidding = 0
        state.pot = 0

    elif action in BETTING_ACTIONS:
        if state.player.stack >= int(action[3:]):
            state.phase = 'BIDDING'
            state.player.action = action
            state.player.action_value = int(action[3:])
            state.player.stack -= int(action[3:])
            state.pot += int(action[3:])

            state.nn_current_bidding += 1

def process_rival_action(state, action, value, next_states):
    new_state = copy_state(state)
    new_state.acting_agent = 'player'

    if action == 'CALL':
        new_state.phase = 'SHOWDOWN'
        new_state.rival.action = action
        new_state.rival.action_value = 5
        new_state.rival.stack -= 5
        new_state.pot += 5

        new_state.showdown()

        new_state.nn_current_hand += 1
        new_state.nn_current_bidding = 0
        new_state.pot = 0

    elif action == 'FOLD':
        new_state.phase = 'SHOWDOWN'
        new_state.rival.action = action
        new_state.player.stack += new_state.pot

        new_state.nn_current_hand += 1
        new_state.nn_current_bidding = 0
        new_state.pot = 0

    elif action + str(value) in BETTING_ACTIONS:
        new_state.phase = 'BIDDING'
        new_state.rival.action = action
        new_state.rival.action_value = value
        new_state.rival.stack -= value
        new_state.pot += value

        new_state.nn_current_bidding += 1

    next_states.append(new_state)


def copy_state(game_state):
    _state = copy.copy(game_state)
    _state.agent = copy.copy(game_state.agent)
    _state.opponent = copy.copy(game_state.opponent)
    return _state

# # Main execution
# if __name__ == "__main__":
#     player = RandomAgent()
#     poker_game = PokerGameSimulation(player)
#     poker_game.play_game()
#     poker_game.display_game_results()
