import math
import random
from matplotlib import pyplot as plt


class Node:
    def __init__(self, id, tp, dem, xx, yy):
        self.id = id
        self.ID = id
        self.type = tp
        self.demand = dem
        self.x = xx
        self.y = yy
        if self.type == 0:
            self.delivery_time = 0
        elif self.type == 1:
            self.delivery_time = 5/60
        elif self.type == 2:
            self.delivery_time = 15/60
        else:
            self.delivery_time = 25/60
        self.scope_radius = 40


class Model:
    def __init__(self):
        self.locations = 200
        self.truck_velocity = 35
        self.capacity = 3000
        self.vehicles = 25
        self.allNodes = []
        self.customers = []
        self.time_matrix = []
        self.dist_matrix = []

    def build_model(self):
        depot = Node(0, 0, 0, 50, 50)
        self.allNodes.append(depot)
        random.seed(1)
        for i in range(0, self.locations):
            id = i + 1
            tp = random.randint(1, 3)
            dem = random.randint(1, 5) * 100
            xx = random.randint(0, 100)
            yy = random.randint(0, 100)
            serv_node = Node(id, tp, dem, xx, yy)
            self.allNodes.append(serv_node)
            self.customers.append(serv_node)
        self.times()

    def distances(self):
        all_nodes = self.allNodes
        dist_matrix = [[0.0 for j in range(0, len(all_nodes))] for k in range(0, len(all_nodes))]
        for i in range(0, len(all_nodes)):
            for j in range(0, i):
                source = all_nodes[i]
                target = all_nodes[j]
                dx_2 = (source.x - target.x) ** 2
                dy_2 = (source.y - target.y) ** 2
                dist = round(math.sqrt(dx_2 + dy_2))
                dist_matrix[i][j] = dist_matrix[j][i] = dist
        self.dist_matrix = dist_matrix

    def times(self):
        all_nodes = self.allNodes
        self.distances()
        time_matrix = [[0.0 for j in range(0, len(all_nodes))] for k in range(0, len(all_nodes))]
        for i in range(1, len(all_nodes)):
            time = self.dist_matrix[0][i] / self.truck_velocity
            time_matrix[0][i] = time + all_nodes[i].delivery_time
        for i in range(0, len(all_nodes)):
            for j in range(1, i):
                time = self.dist_matrix[i][j] / self.truck_velocity
                time_matrix[i][j] = time + all_nodes[j].delivery_time
                time_matrix[j][i] = time + all_nodes[i].delivery_time
        self.time_matrix = time_matrix


class Chain:
    def __init__(self, nodes, time, demand, first_node, last_node):
        self.nodes = nodes
        self.time = time
        self.demand = demand
        self.first_node = first_node
        self.last_node = last_node


class Route:
    def __init__(self, id, nodes, time, demand):
        self.id = id
        self.nodes = nodes
        self.time = time
        self.demand = demand

    def calc_route_cost(self, time_matrix):
        cost = sum([time_matrix[self.nodes[i].id][self.nodes[i + 1].id] for i in range(len(self.nodes) - 1)])
        return cost


class Solution:
    def __init__(self, id, routes, rcl_size=None, seed=None):
        self.id = id
        self.routes = routes
        self.obj = 0 if len(routes) == 0 else max(routes, key=lambda x: x.time).time
        self.rcl_size = rcl_size
        self.seed = seed
        self.median = None
        self.st_dev = None

    def update_median_and_st_dev(self):
        self.median = sum([r.time for r in self.routes]) / len(self.routes)
        var = 0
        for r in self.routes:
            var += (r.time - self.median) ** 2
        var /= len(self.routes) - 1
        self.st_dev = math.sqrt(var)

    def print(self):
        for r in self.routes:
            print(r.id, ":", [n.id for n in r.nodes[:-1]], r.time, r.demand)
        obj = max(self.routes, key=lambda x: x.time)
        print("rcl_size: " + str(self.rcl_size) + " ,seed: " + str(self.seed))
        print("obj=", obj.time)

    def printed_form(self):
        printed_sol = ""
        for r in self.routes:
            printed_nodes = str(r.nodes[0].id)
            for n in r.nodes[1:-1]:
                printed_nodes += "," + str(n.id)
            printed_sol += str(r.id) + ": " + printed_nodes + "   " + str(r.time) + " " + str(r.demand) + "\n"
        printed_sol += "Rcl_size: " + str(self.rcl_size) + " ,seed: " + str(self.seed) + "\n"
        printed_sol += "obj: " + str(self.obj) + "\n\n"
        return printed_sol

    def draw_results(self, name):
        for r in self.routes:
            x_points = [n.x for n in r.nodes[:-1]]
            y_points = [n.y for n in r.nodes[:-1]]
            plt.plot(x_points, y_points, linestyle='dashed', marker='o', markerfacecolor='black', markersize=5)
        plt.savefig(name + ".png")
        plt.close()

    def store_results(self, filename):
        with open(filename, "w") as f:
            f.write(str(self.obj) + "\n")
            for r in self.routes:
                f.write("0")
                for n in r.nodes[1:-1]:
                    f.write("," + str(n.id))
                f.write("\n")
        f.close()
