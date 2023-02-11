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

def heuristic(zone_dict, initial, goal):
    """ This function is used to calculate the heuristic value of the path.

    Args:
        zone_dict (_type_): _description_
        initial (_type_): _description_
        goal (_type_): _description_

    Returns:
        _type_: _description_
    """
    cost = 5
    start_zones = zone_dict[initial]
    goal_zones = zone_dict[goal]
    intersections = start_zones.intersection(goal_zones)
    if len(intersections) == 0:
        cost = cost * 2
    return cost

def ucs(station_dict, initial, goal, zone_dict, algorithm = "BFS", reverse=False):
    """ This function implements the UCS algorithm.

    Args:
        station_dict (_type_): _description_
        initial (_type_): _description_
        goal (_type_): _description_
        zone_dict (_type_): _description_
        algorithm (_type_, optional): _description_. Defaults to "BFS".
        reverse (_type_, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    init_cost = 0
    number_of_explored_nodes = 0

    if algorithm == "BFS":
        init_cost = heuristic(zone_dict, initial, goal)

    visited_nodes = {}
    visited_nodes[initial] = (init_cost, [initial], 0)

    explore_path = PriorityQueue()
    explore_path.put((init_cost, [initial], 0))

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

            h = heuristic(zone_dict,child_label,goal)
            a_total_cost_to_neighbor = total_cost + cost_to_neighbor
            if algorithm == "BFS":
                total_cost_to_neighbor = h
            else:
                total_cost_to_neighbor = a_total_cost_to_neighbor
            if (child_label not in visited_nodes) or (visited_nodes[child_label][0] > total_cost_to_neighbor):
                next_node = (total_cost_to_neighbor, path + [child_label], a_total_cost_to_neighbor)
                visited_nodes[child_label] = next_node
                explore_path.put(next_node)
                number_of_explored_nodes += 1
    return visited_nodes[goal], number_of_explored_nodes

start = 'Canada Water'
end = 'Stratford'
reverse = False

ucs_solution, visited_ucs = ucs(station_dict, start, end, zone_dict)
print("UCS")
print(ucs_solution[1])
print("UCS visited nodes: {}".format(visited_ucs))
print("UCS cost: {}".format(ucs_solution[2]))
print("Solution length {}".format(len(ucs_solution[1])))