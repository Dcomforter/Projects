from pyamaze import maze, agent, textLabel, COLOR
from queue import PriorityQueue
#from types import NoneType
import types
import tracemalloc
import time


class Node:
    def __init__(self, position, parent, g):
        self.position = position
        self.parent = parent
        self.g = g
        self.h = 0

    def __lt__(self, other):
        return self.h < other.h


class IDANode:
    def __init__(self, position, g):
        self.position = position
        self.g = g

    def __lt__(self, other):
        return self.h < other.h


def h(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def a_star(m, start, goal):
    start_node = Node(start, None, 0)
    start_node.h = h(start, goal)
    fringe = PriorityQueue()
    priority = start_node.h + start_node.g
    fringe.put((priority, start_node))
    closed = []
    searched = [start]
    while not fringe.empty():
        current = fringe.get()[1]
        searched.append(current.position)
        if current.position in closed:
            continue
        else:
            closed.append(current.position)
        if current.position == goal:
            break
        neighbor_list = [(current.position[0] + 1, current.position[1]), (current.position[0] - 1, current.position[1]),
                         (current.position[0], current.position[1] + 1), (current.position[0], current.position[1] - 1)]
        for i in "NESW":
            if m.maze_map[current.position][i] == True:
                if i == "N":
                    temp_position = (current.position[0] - 1, current.position[1])
                    tempNode = Node(temp_position, current, current.g + 1)
                    if temp_position[0] < 1:
                        continue
                if i == "E":
                    temp_position = (current.position[0], current.position[1] + 1)
                    tempNode = Node(temp_position, current, current.g + 1)
                    if temp_position[1] > m.rows:
                        continue
                if i == "S":
                    temp_position = (current.position[0] + 1, current.position[1])
                    tempNode = Node(temp_position, current, current.g + 1)
                    if temp_position[0] > m.cols:
                        continue
                if i == "W":
                    temp_position = (current.position[0], current.position[1] - 1)
                    tempNode = Node(temp_position, current, current.g + 1)
                    if temp_position[1] < 1:
                        continue
                if temp_position in closed:
                    continue
                tempNode.h = h(tempNode.position, goal)
                priority = tempNode.h + tempNode.g
                fringe.put((priority, tempNode))

    path = []
    NoneType = type(None)
    while type(current) != NoneType:
        path.append(current.position)
        current = current.parent
    return path[::-1], searched


def search(m, node, threshold, goal, path):
    if node.position == goal:
        return node
    f = node.g + h(node.position, goal)
    if (f > threshold):
        return f
    min = float("inf")
    tempNode = None
    for i in "NESW":
        if m.maze_map[node.position][i] == True:
            if i == "N":
                temp_position = (node.position[0] - 1, node.position[1])
                tempNode = IDANode(temp_position, node.g + 1)
                if temp_position[0] < 1:
                    continue
            if i == "E":
                temp_position = (node.position[0], node.position[1] + 1)
                tempNode = IDANode(temp_position, node.g + 1)
                if temp_position[1] > m.rows:
                    continue
            if i == "S":
                temp_position = (node.position[0] + 1, node.position[1])
                tempNode = IDANode(temp_position, node.g + 1)
                if temp_position[0] > m.cols:
                    continue
            if i == "W":
                temp_position = (node.position[0], node.position[1] - 1)
                tempNode = IDANode(temp_position, node.g + 1)
                if temp_position[1] < 1:
                    continue
            if (tempNode.position not in path):
                path.append(tempNode.position)
                temp = search(m, tempNode, threshold, goal, path)
                if type(temp) == IDANode:
                    if (temp.position == goal):
                        return temp
                if temp < min:
                    min = temp
                path.pop()
    return min


def ida_star(m, start, goal):
    path = [start]
    start = IDANode(start, 0)
    threshold = h(start.position, goal)
    while (1):
        temp = search(m, start, threshold, goal, path)
        if (type(temp) == IDANode):
            return path
        if (temp == float("inf")):
            return -1
        threshold = temp


if __name__ == "__main__":
    m = maze(50, 50)
    print("Maze: 50x50")
    m.CreateMaze(loopPercent=50)
    start = (1, 1)
    goal = (m.rows, m.cols)

    tracemalloc.start()
    starttime = time.perf_counter()
    path, searched = a_star(m, goal, start)
    endtime = time.perf_counter()
    print("Peak Memory", tracemalloc.get_traced_memory()[1] / (1024), "KB")
    tracemalloc.stop()
    print("A star total time:", endtime - starttime)
    a = agent(m, filled=True, footprints=True)
    b = agent(m, footprints=True, color=COLOR.light, filled=True)
    c = agent(m, footprints=True, color=COLOR.yellow, filled=True)
    m.tracePath({b: searched}, delay=10)
    m.tracePath({a: path}, delay=10)
    m.tracePath({c: m.path}, delay=10)

    tracemalloc.start()
    starttime = time.perf_counter()
    path = ida_star(m, goal, start)
    endtime = time.perf_counter()
    print("Peak Memory", tracemalloc.get_traced_memory()[1] / (1024), "KB")
    tracemalloc.stop()
    print("IDA star total time:", endtime - starttime)

    d = agent(m, filled=True, footprints=True, color=COLOR.red)
    m.tracePath({d: path}, delay=10)
    l = textLabel(m, 'A Star', len(path) + 1)
    l2 = textLabel(m, 'Built in A Star', len(m.path) + 1)
    l3 = textLabel(m, 'IDA Star', len(path) + 1)
    m.run()


