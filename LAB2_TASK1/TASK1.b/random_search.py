# How This Code Works:
# generateMap2d_obstacle: Generates a 2D grid map with a 'rotated-H' shaped obstacle.
# get_neighbors: Finds accessible neighboring cells for a given position on the grid.
# random_search: Performs a random search from the start position to the goal. It randomly selects a neighboring cell to move to at each step.
# plotMap: Visualizes the map and the path found by the Random Search algorithm.


import numpy as np
import matplotlib.pyplot as plt
import random

def generateMap2d_obstacle(size, obstacle_y, obstacle_x, start=(0, 0), goal=(9, 9)):
    map2d = np.zeros(size)
    map2d[obstacle_y[0], :] = -1  # Top horizontal line of 'H'
    map2d[obstacle_y[1], :] = -1  # Bottom horizontal line of 'H'
    map2d[obstacle_y[0]:obstacle_y[1] + 1, obstacle_x] = -1  # Vertical line of 'H'
    map2d[start] = 0
    map2d[goal] = 0
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

def random_search(map2d, start, goal, max_iterations=10000):
    current = start
    path = [current]
    visited = set([current])

    for _ in range(max_iterations):
        if current == goal:
            return path
        neighbors = get_neighbors(current, map2d)
        if not neighbors:
            break  # No more moves possible
        current = random.choice(neighbors)
        path.append(current)
        visited.add(current)

    return path  # Return the path if goal is not found within the iteration limit

def plotMap(map2d, path, title="Random Search Path"):
    plt.imshow(map2d, cmap='gray_r')
    plt.title(title)
    if path:
        x_path, y_path = zip(*path)
        plt.plot(y_path, x_path, 'b-', label='Path')
        plt.plot(path[0][1], path[0][0], 'go', label='Start')  # Start in green
        plt.plot(path[-1][1], path[-1][0], 'ro', label='Goal')  # Goal in red
        plt.legend()
    plt.show()

# Parameters for map generation and search
map_size = (10, 10)
obstacle_y = (3, 6)
obstacle_x = 5
start = (0, 0)
goal = (9, 9)

# Generate the map with the 'rotated-H' obstacle
map_with_obstacle = generateMap2d_obstacle(map_size, obstacle_y, obstacle_x, start, goal)

# Run Random Search
path = random_search(map_with_obstacle, start, goal)
print("Found Path:", path)

# Plot the path
plotMap(map_with_obstacle, path)
