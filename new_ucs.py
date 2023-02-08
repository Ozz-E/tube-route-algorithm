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
    line = row[2]

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

def nucs(station_dict, initial, goal, zone_dict, line_change_time, reverse=False):
    init_cost = 0
    number_of_explored_nodes = 0

    visited_nodes = {}
    visited_nodes[initial] = (init_cost, [initial], init_cost)

    explore_path = PriorityQueue()
    explore_path.put((init_cost, [initial], init_cost))

    while not explore_path.empty():
        cost, path, total_time = explore_path.get()

        if path[-1] == goal:
            return path, total_time, number_of_explored_nodes

        for neighbor, neighbor_cost in station_dict[path[-1]]:
            new_cost = cost + neighbor_cost
            new_time = total_time + neighbor_cost

            if neighbor in path:
                continue

            line_change = False
            if neighbor in zone_dict and path[-1] in zone_dict:
                if len(zone_dict[neighbor].intersection(zone_dict[path[-1]])) == 0:
                    line_change = True
                    new_time += line_change_time

            if neighbor in visited_nodes:
                old_cost, _, _ = visited_nodes[neighbor]
                if old_cost <= new_cost:
                    continue

            number_of_explored_nodes += 1
            visited_nodes[neighbor] = (new_cost, path + [neighbor], new_time)
            explore_path.put((new_cost, path + [neighbor], new_time))

    return number_of_explored_nodes, new_time, visited_nodes,


initial = 'Canada Water'
goal = 'Stratford'
line_change_time = 2

#initial = 'New Cross Gate'
#goal =  'Stepney Green'

#initial = 'Ealing Broadway'
#goal = 'South Kensington'

#initial = 'Baker Street'
#goal = 'Wembley Park'

ucs_solution, ucs_time, visited_ucs = nucs(station_dict, initial, goal, zone_dict, line_change_time)
print(ucs_solution)
print("UFS visited nodes: {}".format(visited_ucs))
print("UFS time: {}".format(ucs_time))
print("Solution length {}".format(len(ucs_solution[1])))

