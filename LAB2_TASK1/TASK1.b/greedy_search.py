import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue
import matplotlib.cm as cm

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

def generateMap2d(size_):
    size_x, size_y = size_[0], size_[1]
    map2d = np.random.rand(size_y, size_x)
    perObstacles_ = 0.01  # 10% of the area as obstacles
    map2d[map2d <= perObstacles_] = -1  # Obstacles
    map2d[map2d > perObstacles_] = 0   # Free space
    return map2d

def generateMap2d_obstacle(size_):
    size_x, size_y = size_[0], size_[1]
    map2d = generateMap2d(size_)

    # Adding special obstacle (rotated-H)
    xtop = [np.random.randint(5, 3 * size_x // 10 - 2), np.random.randint(7 * size_x // 10 + 3, size_x - 5)]
    ytop = np.random.randint(7 * size_y // 10 + 3, size_y - 5)
    xbot = [np.random.randint(3, 3 * size_x // 10 - 5), np.random.randint(7 * size_x // 10 + 3, size_x - 5)]
    ybot = np.random.randint(5, size_y // 5 - 3)

    map2d[ybot, xbot[0]:xbot[1] + 1] = -1
    map2d[ytop, xtop[0]:xtop[1] + 1] = -1
    minx = (xbot[0] + xbot[1]) // 2
    maxx = (xtop[0] + xtop[1]) // 2
    if minx > maxx:
        minx, maxx = maxx, minx
    if maxx == minx:
        maxx += 1

    map2d[ybot:ytop, minx:maxx] = -1

    # Ensuring start and goal are not placed on an obstacle
    while True:
        startp = [np.random.randint(0, size_x // 2 - 4), np.random.randint(5, size_y - 5)]
        goalp = [np.random.randint(size_x // 2 + 4, size_x - 1), np.random.randint(5, size_y - 5)]
        if map2d[startp[0], startp[1]] == 0 and map2d[goalp[0], goalp[1]] == 0:
            map2d[startp[0], startp[1]] = -2  # Start point
            map2d[goalp[0], goalp[1]] = -3  # Goal point
            break

    return map2d, startp, goalp

def get_neighbors(node, map2d):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    neighbors = []
    x, y = node
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        # Check for valid neighbor within map bounds and not an obstacle
        if 0 <= nx < map2d.shape[0] and 0 <= ny < map2d.shape[1] and map2d[nx, ny] != -1:
            neighbors.append((nx, ny))
    return neighbors

def manhattan_distance(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

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

def plotMap(map2d_, path_, title_='DFS Search Path'):
    plt.interactive(False)

    greennumber = int(map2d_.max() + 1)
    colors = cm.winter(np.linspace(0, 1, greennumber))

    colorsMap2d = np.ones((map2d_.shape[0], map2d_.shape[1], 4))  # RGBA format

    locStart = np.where(map2d_ == -2)
    locEnd = np.where(map2d_ == -3)
    colorsMap2d[locStart[0], locStart[1]] = [0.0, 1.0, 0.0, 1.0]  # Green
    colorsMap2d[locEnd[0], locEnd[1]] = [0.0, 1.0, 1.0, 1.0]  # Cyan

    locObstacle = np.where(map2d_ == -1)
    colorsMap2d[locObstacle[0], locObstacle[1]] = [1.0, 0.0, 0.0, 1.0]  # Red

    locExpand = np.where(map2d_ > 0)
    for iposExpand in range(len(locExpand[0])):
        colorsMap2d[locExpand[0][iposExpand], locExpand[1][iposExpand]] = colors[int(map2d_[locExpand[0][iposExpand], locExpand[1][iposExpand]] - 1)]

    plt.figure()
    plt.title(title_)
    plt.imshow(colorsMap2d, interpolation='nearest')
    plt.colorbar()
    plt.plot([p[1] for p in path_], [p[0] for p in path_], color='magenta', linewidth=2.5)
    plt.ylim(0, map2d_.shape[0])
    plt.xlim(0, map2d_.shape[1])
    plt.show()

# Testing the DFS Search Algorithm
map_size = (100, 100)
map_with_obstacle, start, goal = generateMap2d_obstacle(map_size)


# Run BFS
path = greedy_search(map_with_obstacle, tuple(start), tuple(goal))

# Print start, goal, and path for debugging
print(f"Start: {start}, Goal: {goal}")
print(f"Path: {path}")


# Plot the path
plotMap(map_with_obstacle, path)  # Pass the path directly