# A* algorithm with Manhattan distance as the heuristic is a great choice for grid-based pathfinding where movements are restricted to horizontal and vertical steps. Manhattan distance calculates the sum of the absolute differences of the coordinates of two points, which aligns well with grid movement.

# Manhattan Distance Heuristic: This heuristic is particularly effective for grid-based environments, where diagonal movement is not allowed. It calculates the total number of steps moved horizontally and vertically to reach the goal from the current position.

# Optimality and Completeness: A* with Manhattan distance is both complete and optimal in grid environments without diagonal movement. It will find the shortest path, assuming the heuristic is admissible.

# Efficiency: The efficiency of A*

import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue

class Node:
    def __init__(self, position, parent=None, g=0, h=0):
        self.position = position
        self.parent = parent
        self.g = g  # Cost from start to current node
        self.h = h  # Heuristic cost to goal

    def total_cost(self):
        return self.g + self.h

    def __lt__(self, other):
        return self.total_cost() < other.total_cost()

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

def a_star_search(map2d, start, goal):
    start_node = Node(start, g=0, h=manhattan_distance(start, goal))
    goal_node = Node(goal)
    open_set = PriorityQueue()
    open_set.put((start_node.total_cost(), start_node))
    visited = set()

    while not open_set.empty():
        _, current_node = open_set.get()

        if current_node.position == goal_node.position:
            return current_node.get_path()

        visited.add(current_node.position)

        for neighbor in get_neighbors(current_node.position, map2d):
            if neighbor in visited:
                continue

            neighbor_node = Node(neighbor, current_node, current_node.g + 1, manhattan_distance(neighbor, goal))

            if all(neighbor != node.position for _, node in open_set.queue):
                open_set.put((neighbor_node.total_cost(), neighbor_node))

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

# Testing the A* Search Algorithm with Manhattan Distance
map_size = (10, 10)
start = (0, 0)
goal = (9, 9)
map2d = generateMap2d(map_size)
path = a_star_search(map2d, start, goal)
print("Path:", path)
plotMap(map2d, path, "A* Search Path with Manhattan Distance")
