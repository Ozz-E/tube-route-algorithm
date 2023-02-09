import csv
import networkx as nx
from queue import PriorityQueue

G = nx.Graph()

with open('tubedata.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader) # skip the header
    for row in reader:
        start_station = row[0]
        end_station = row[1]
        cost = int(row[3])
        zone_one = int(row[4])
        zone_two = int(row[5])
        if int(row[5]) != 0:
            G.add_edge(start_station, end_station, weight=cost,zone1=zone_one, zone2=zone_two)
        else:
            G.add_edge(start_station, end_station, weight=cost,zone1=zone_one, zone2=zone_one)


def heuristic(G, initial, goal):
    cost = 5
    print(nx.get_edge_attributes(G, 'zone1')[(initial, goal)])
    start_station = initial.strip('')
    print(nx.get_node_attributes(G,'zone1')[start_station])
    start_zones = nx.get_node_attributes(G, 'zone1')[initial]
    goal_zones = nx.get_node_attributes(G, 'zone1')[goal]
    intersections = start_zones.intersection(goal_zones)
    if len(intersections) == 0:
        cost = cost * 2
    return cost


def ucs(initial, goal, G, algorithm = "BFS", reverse=False):
    init_cost = 0
    number_of_explored_nodes = 0
    if algorithm == "BFS":
        init_cost = heuristic(G, initial, goal)

    visited_nodes = {}
    visited_nodes[initial] = (init_cost, [initial],0)

    explore_path = PriorityQueue()
    explore_path.put((init_cost, [initial],0))
    while not explore_path.empty():

        _, path, total_cost = explore_path.get()
        
        node = path[-1]

        if node == goal:
            return visited_nodes[node], number_of_explored_nodes
        
        neighbors = G.neighbors(node)

        if reverse:
            neighbors = reversed(neighbors)

            for child in neighbors:
                cost_to_neighbor = G[node][child]['weight']
                h = heuristic(G,child,goal)
                a_total_cost_to_neighbor = total_cost + cost_to_neighbor
                if algorithm == "BFS":
                    total_cost_to_neighbor = h
                else:
                    total_cost_to_neighbor = a_total_cost_to_neighbor
                if (child not in visited_nodes) or (visited_nodes[child][0] > total_cost_to_neighbor):
                    next_node = (total_cost_to_neighbor, path + [child], a_total_cost_to_neighbor)
                    visited_nodes[child] = next_node
                    explore_path.put(next_node)
                    number_of_explored_nodes += 1
    return visited_nodes, number_of_explored_nodes

start = "Canada Water"
end = "Stratford"
reverse = False

ucs_solution, visited_ucs = ucs(start, end, G)
print("UCS")
print(ucs_solution)
print(ucs_solution[1])
#print("UCS visited nodes: {}".format(visited_ucs))
#print("UCS cost: {}".format(ucs_solution[2]))
print("Solution length {}".format(len(ucs_solution[1])))
