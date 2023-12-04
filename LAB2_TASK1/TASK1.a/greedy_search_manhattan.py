import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue

class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent

    def get_path(self):
        path, current_node = [], self
        while current_node:
            path.append(current_node.position)
            current_node = current_node.parent
        return path[::-1]

def manhattan_distance(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

def generateMap2d(size, percentOfObstacle=0.1, start=(0, 0), goal=(9, 9)):
    map2d = np.zeros(size)
    num_obstacles = int(np.prod(size) * percentOfObstacle)
    obstacles = set()

    while len(obstacles) < num_obstacles:
        obstacle = (np.random.randint(0, size[0]), np.random.randint(0, size[1]))
        if obstacle != start and obstacle != goal:
            map2d[obstacle] = -1
            obstacles.add(obstacle)

    return map2d

def get_neighbors(node, map2d):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    neighbors = []
    x, y = node

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < map2d.shape[0] and 0 <= ny < map2d.shape[1] and map2d[nx, ny] != -1:
            neighbors.append((nx, ny))

    return neighbors

# Manhattan distance is a common heuristic for grid-based pathfinding, particularly when movement is restricted to horizontal and vertical directions (like on a typical 2D grid map). It calculates the absolute sum of the horizontal and vertical distances between two points, which aligns well with grid-based movement.

# Manhattan Distance: This heuristic is particularly effective in grid environments where movement is limited to horizontal and vertical directions. It calculates the total number of horizontal and vertical steps to reach the goal from the current position.

# Efficiency: The Manhattan distance heuristic aligns well with the nature of grid-based movement, potentially leading to a more direct path to the goal compared to Euclidean distance in certain scenarios, especially when diagonal movement is not allowed.

# Optimality: Similar to the Euclidean distance heuristic, the Greedy Search with Manhattan distance does not guarantee the shortest path. It chooses the next step based solely on the heuristic, which can sometimes lead to suboptimal paths.

# Use Case: It's particularly useful in urban-like grid layouts or in scenarios where diagonal movement is either not possible or not preferable.


def greedy_search(map2d, start, goal):
    start_node = Node(start)
    goal_node = Node(goal)
    open_set = PriorityQueue()
    open_set.put((0, 0, start_node))
    visited = set()
    counter = 1

    while not open_set.empty():
        _, _, current_node = open_set.get()

        if current_node.position == goal_node.position:
            return current_node.get_path()

        visited.add(current_node.position)

        for neighbor in get_neighbors(current_node.position, map2d):
            if neighbor not in visited:
                neighbor_node = Node(neighbor, current_node)
                heuristic_cost = manhattan_distance(neighbor, goal)
                open_set.put((heuristic_cost, counter, neighbor_node))
                counter += 1

    return []  # No path found

def plotMap(map2d, path, title="Path"):
    plt.imshow(map2d, cmap='gray_r')
    plt.title(title)

    if path:
        x_path, y_path = zip(*path)
        start, goal = path[0], path[-1]
        plt.plot(start[1], start[0], 'go')
        plt.plot(goal[1], goal[0], 'ro')
        plt.plot(y_path, x_path, 'b-')
    else:
        print("No path to plot.")

    plt.show()

# Testing the Greedy Search Algorithm with Manhattan Distance
map_size = (10, 10)
start = (0, 0)
goal = (9, 9)
map2d = generateMap2d(map_size)
path = greedy_search(map2d, start, goal)
print("Path:", path)
plotMap(map2d, path, "Greedy Search Path with Manhattan Distance")
