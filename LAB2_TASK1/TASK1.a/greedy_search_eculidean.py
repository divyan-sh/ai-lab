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

def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

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

def greedy_search(map2d, start, goal):
    start_node = Node(start)
    goal_node = Node(goal)
    open_set = PriorityQueue()
    open_set.put((0, 0, start_node))  # Added a tie-breaker (second element in the tuple)
    visited = set()
    counter = 1  # Counter for tie-breaking

    while not open_set.empty():
        _, _, current_node = open_set.get()

        if current_node.position == goal_node.position:
            return current_node.get_path()

        visited.add(current_node.position)

        for neighbor in get_neighbors(current_node.position, map2d):
            if neighbor not in visited:
                neighbor_node = Node(neighbor, current_node)
                heuristic_cost = euclidean_distance(neighbor, goal)
                open_set.put((heuristic_cost, counter, neighbor_node))
                counter += 1

    return []  # No path found

def plotMap(map2d, path, title="Path"):
    plt.imshow(map2d, cmap='gray_r')
    plt.title(title)

    if path:
        x_path, y_path = zip(*path)
        start, goal = path[0], path[-1]
        plt.plot(start[1], start[0], 'go')  # Start in green
        plt.plot(goal[1], goal[0], 'ro')   # Goal in red
        plt.plot(y_path, x_path, 'b-')  # Path in blue line
    else:
        print("No path to plot.")

    plt.show()

# Testing the Greedy Search Algorithm
map_size = (10, 10)
start = (0, 0)
goal = (9, 9)
map2d = generateMap2d(map_size)
path = greedy_search(map2d, start, goal)
print("Path:", path)
plotMap(map2d, path, "Greedy Search Path")
