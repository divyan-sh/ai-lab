from pandas import DataFrame
from poker_game import PokerGameSimulation
from randomAgent import RandomAgent
from breadthFirstSearchAgent import BreadthFirstAgent
from depthFirstSearchAgent import DepthFirstAgent

def run_game(agent):
    poker_game = PokerGameSimulation(agent)
    final_stack, hands_played = poker_game.play_game()
    return final_stack, hands_played

def analyze_results(resultList):
    analysedResults = {"AgentType":resultList[0]["AgentType"]
        , "AgentStack":0,"OpponentStack":0 ,"PathLength":0 ,"Expanded": 0,"Wins":0, "Hands": 0}
    for result in resultList:
        for key in result.keys():
            if key != "AgentType":
                analysedResults[key] += result[key]
                
        if result["AgentStack"] > result["OpponentStack"]:
            analysedResults["Wins"] += 1

    for key in analysedResults.keys():
        if key != "AgentType" and key != "Wins":
            analysedResults[key] = analysedResults[key]/len(resultList)
    return analysedResults

# List of different agent instances
agents = [RandomAgent(stack=400),BreadthFirstAgent(400),DepthFirstAgent(stack=400)]

result_data = []
for agent in agents:
    agent_results = []
    for _ in range(10):
        final_stack, hands_played = run_game(agent)
        agent_results.append({'final_stack': final_stack, 'hands_played': hands_played})

    analyzed_results = analyze_results(agent_results)
    result_data.append(analyzed_results)

# Create and display results DataFrame
result_df = DataFrame(result_data)
print(result_df)
