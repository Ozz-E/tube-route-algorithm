import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from queue import PriorityQueue
from collections import defaultdict
import csv

# Read data from csv file and convert it to dictionary, add attributes to the dictionary
df = pd.read_csv('tubedata.csv', header=None)
# no header -> can just assign new column names
df.head()

station_dict = defaultdict(list)
zone_dict = defaultdict(set)

for index, row in df.iterrows():

    start_station = row[0]
    end_station = row[1]
    cost = int(row[3])

    zone1 = row[4]
    zone2 = row[5]

    station_list = station_dict[start_station]
    station_list.append((end_station, cost))

    station_list = station_dict[end_station]
    station_list.append((start_station, cost))

    zone_dict[start_station].add(zone1)
    if zone2 != "0":
        zone_dict[start_station].add(zone2)
        zone_dict[end_station].add(zone1)
    else:
        zone_dict[end_station].add(zone1)


# The following section of the code: construct_path_from_root(), DFS and BFS have been taken from ECS759P Lab3 exercise


def construct_path_from_root(node, root):
    path_from_root = [node['label']]
    while node['parent']:
        node = node['parent']
        path_from_root = [node['label']] + path_from_root
    return path_from_root


def get_path_cost(solution_path, station_dict):
    total_cost = 0
    for i in range(len(solution_path) - 1):

        initial = solution_path[i]
        destination = solution_path[i + 1]
        station_list = station_dict[initial]

        for station in station_list:
            if station[0] == destination:
                total_cost += station[1]
                break
    return total_cost


def dfs(station_dict, initial, goal, reverse=False):
    frontier = [{'label': initial, 'parent': None}]
    explored = {initial}
    number_of_explored_nodes = 0

    while frontier:
        node = frontier.pop()  # pop from the right of the list
        number_of_explored_nodes += 1
        if node['label'] == goal:
            return node, number_of_explored_nodes

        neighbours = reversed(list(station_dict[node['label']])) if reverse else station_dict[node['label']]

        for child in neighbours:
            child_label = child[0]
            child = {'label': child_label, 'parent': node}
            if child_label not in explored:
                frontier.append(child)  # added to the right of the list, so it is a LIFO
                explored.add(child_label)
    return None, number_of_explored_nodes


def bfs(station_dict, initial, goal, reverse=False):
    number_of_explored_nodes = 1
    frontier = [{'label': initial, 'parent': None}]
    # FIFO queue implementation with a list is slow. For bigger problems, better to use deque.
    explored = {initial}

    if initial == goal:  # just in case, because now we are checking the children
        return None

    while frontier:
        node = frontier.pop()  # pop from the right of the list

        neighbours = reversed(list(station_dict[node['label']])) if reverse else station_dict[node['label']]

        for child in neighbours:
            child_label = child[0]
            child = {'label': child_label, 'parent': node}
            if child_label == goal:
                return child, number_of_explored_nodes

            if child_label not in explored:
                frontier = [child] + frontier  # added to the left of the list, so a FIFO
                number_of_explored_nodes += 1
                explored.add(child_label)

    return None, number_of_explored_nodes


def ucs(station_dict, initial, goal, zone_dict, reverse=False):

    init_cost = 0
    number_of_explored_nodes = 0

    visited_nodes = {}
    visited_nodes[initial] = (init_cost, [initial], init_cost)

    explore_path = PriorityQueue()
    explore_path.put((init_cost, [initial], init_cost))

    while not explore_path.empty():

        _, path, total_cost = explore_path.get()

        node = path[-1]

        if node == goal:
            return visited_nodes[goal], number_of_explored_nodes

        neighbors = station_dict[node]

        if reverse:
            neighbors = reversed(neighbors)

        for child in neighbors:
            child_label = child[0]
            cost_to_neighbor = child[1]
            total_cost_to_neighbor = total_cost + cost_to_neighbor
            if (child_label not in visited_nodes) or (visited_nodes[child_label][0] > total_cost_to_neighbor):
                next_node = (total_cost_to_neighbor, path + [child_label], total_cost_to_neighbor)
                visited_nodes[child_label] = next_node
                explore_path.put(next_node)
                number_of_explored_nodes += 1
    return visited_nodes[goal], number_of_explored_nodes


start = 'Canada Water'
end = 'Stratford'
reverse = False
# pre-set for other explored paths

# start = 'New Cross Gate'
# end =  'Stepney Green'

# start = 'Ealing Broadway'
# end = 'South Kensington'

# start = 'Baker Street'
# end = 'Wembley Park'


dfs_solution, visited_dfs = dfs(station_dict, start, end, reverse=reverse)
dfs_path = construct_path_from_root(dfs_solution, start)
print("DFS")
print(dfs_path)
print("DFS visited nodes: {}".format(visited_dfs))
print("DFS cost: {}".format(get_path_cost(dfs_path, station_dict)))
print("Solution length {}".format(len(dfs_path)))

bfs_solution, visited_bfs = bfs(station_dict, start, end)
bfs_path = construct_path_from_root(bfs_solution, start)
print("BFS")
print(bfs_path)
print("BFS visited nodes: {}".format(visited_bfs))
print("BFS cost: {}".format(get_path_cost(bfs_path, station_dict)))
print("Solution length {}".format(len(bfs_path)))

ucs_solution, visited_ucs = ucs(station_dict, start, end, zone_dict)
print("UCS")
print(ucs_solution[1])
print("UFS visited nodes: {}".format(visited_ucs))
print("UFS cost: {}".format(ucs_solution[2]))
print("Solution length {}".format(len(ucs_solution[1])))

reverse = True

dfs_solution, visited_dfs = dfs(station_dict, start, end, reverse=reverse)
dfs_path = construct_path_from_root(dfs_solution, start)
print("DFS")
print(dfs_path)
print("DFS visited nodes: {}".format(visited_dfs))
print("DFS cost: {}".format(get_path_cost(dfs_path, station_dict)))
print("Solution length {}".format(len(dfs_path)))

bfs_solution, visited_bfs = bfs(station_dict, start, end)
bfs_path = construct_path_from_root(bfs_solution, start)
print("BFS")
print(bfs_path)
print("BFS visited nodes: {}".format(visited_bfs))
print("BFS cost: {}".format(get_path_cost(bfs_path, station_dict)))
print("Solution length {}".format(len(bfs_path)))

ucs_solution, visited_ucs = ucs(station_dict, start, end, zone_dict)
print("UCS")
print(ucs_solution[1])
print("UCS visited nodes: {}".format(visited_ucs))
print("UCS cost: {}".format(ucs_solution[2]))
print("Solution length {}".format(len(ucs_solution[1])))
