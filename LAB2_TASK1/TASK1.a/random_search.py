import numpy as np
import random
import matplotlib.pyplot as plt

# Function to generate a 2D map with obstacles
def generateMap2d(size, percentOfObstacle=0.3, start=(0, 0), goal=(9, 9)):
    map2d = np.ones(size)
    num_obstacles = int(np.prod(size) * percentOfObstacle)
    obstacles = set()

    while len(obstacles) < num_obstacles:
        obstacle = (random.randint(0, size[0]-1), random.randint(0, size[1]-1))
        if obstacle not in obstacles and obstacle != start and obstacle != goal:
            map2d[obstacle] = -1  # Marking the obstacle
            obstacles.add(obstacle)

    # Mark start and goal on the map
    map2d[start] = -2
    map2d[goal] = -3

    return map2d

# Function to visualize the path on the map
def plotMap(map2d, path, title="Path"):
    plt.imshow(map2d, cmap='gray_r')
    plt.title(title)

    # Extracting x and y coordinates from the path
    x_path, y_path = zip(*path)

    # Marking the start and goal points
    start, goal = path[0], path[-1]
    print(goal)
    plt.plot(start[1], start[0], 'go')  # Start in green
    plt.plot(goal[1], goal[0], 'ro')   # Goal in red

    # Plotting the path with a line
    plt.plot(y_path, x_path, 'b-')  # Path in blue line

    plt.show()

# Random Search Algorithm
def random_search(map2d, start, goal):
    max_iterations = 10000
    current = start
    path = [current]
    visited = set()

    for _ in range(max_iterations):
        if current == goal:
            return path

        visited.add(current)
        neighbors = get_neighbors(current, map2d)
        random_neighbor = random.choice(neighbors) if neighbors else None

        if random_neighbor and random_neighbor not in visited:
            current = random_neighbor
            path.append(current)

    return path

# Helper function to get neighbors of a node
def get_neighbors(node, map2d):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
    neighbors = []

    for dx, dy in directions:
        x, y = node[0] + dx, node[1] + dy
        if 0 <= x < map2d.shape[0] and 0 <= y < map2d.shape[1] and map2d[x, y] >= 0:
            neighbors.append((x, y))

    return neighbors

# Test the Random Search Algorithm
map_size = (10, 10)
start = (0, 0)
goal = (9, 9)
map2d = generateMap2d(map_size, percentOfObstacle=0.3)
path = random_search(map2d, start, goal)
print(path)
plotMap(map2d, path, "Random Search Path")
