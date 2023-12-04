import numpy as np
import matplotlib.pyplot as plt
from queue import Queue

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

def generateMap2d(size, percentOfObstacle=0.1, start=(0, 0), goal=(9, 9)):
    map2d = np.zeros(size)
    num_obstacles = int(np.prod(size) * percentOfObstacle)
    obstacles = set()

    while len(obstacles) < num_obstacles:
        obstacle = (np.random.randint(0, size[0]), np.random.randint(0, size[1]))
        if obstacle != start and obstacle != goal:
            map2d[obstacle] = -1  # Marking the obstacle
            obstacles.add(obstacle)

    return map2d

def get_neighbors(node, map2d):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
    neighbors = []
    x, y = node

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < map2d.shape[0] and 0 <= ny < map2d.shape[1] and map2d[nx, ny] != -1:
            neighbors.append((nx, ny))

    return neighbors

def bfs_search(map2d, start, goal):
    start_node = Node(start)
    goal_node = Node(goal)
    queue = Queue()
    queue.put(start_node)
    visited = set([start])

    while not queue.empty():
        current_node = queue.get()

        if current_node.position == goal_node.position:
            return current_node.get_path()

        for neighbor in get_neighbors(current_node.position, map2d):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.put(Node(neighbor, current_node))

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

# Testing the BFS Search Algorithm
map_size = (10, 10)
start = (0, 0)
goal = (9, 9)
map2d = generateMap2d(map_size)
path = bfs_search(map2d, start, goal)
print("Path:", path)
plotMap(map2d, path, "BFS Search Path")
