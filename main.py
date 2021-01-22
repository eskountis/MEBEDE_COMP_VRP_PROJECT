import random
import time

from Competitional import FileHandler, Implementation, Starting_Sol, VRP_Model, Search
from Competitional.Model import Model, Solution

m = Model()
m.build_model()
locations = m.customers
time_matrix = m.time_matrix
depot = m.allNodes[0]

# sort_by_distance_from_depot = sorted(locations, key=lambda x: time_matrix[0][x.id], reverse=True)
# sort_by_demands = sorted(locations, key=lambda x: x.demand, reverse=True)
# filename = "Good_Solutions.txt"
# sorted_nodes = sort_by_distance_from_depot
# # signature = "_by_demands"
# signature = "_by_distance"
radius = 20
filename = "Award Winning Solutions.txt"
signature = "_by_clusters"

rcl_sizes = [5, 7, 10]
tabu_list_sizes = [20, 30]
probs = [0.55, 0.6]
for tabu_size in tabu_list_sizes:
    for rcl_size in rcl_sizes:
        for prob in probs:
            for seed in range(101, 111):
                if rcl_size == 5 and tabu_size == 20 and prob == 0.55:
                    continue
                sol = Starting_Sol.route_clustering_with_tsp_nearest(radius, m)
                print(sol.obj)
                # sol.draw_results("before")
                # sol.print()
                # Search.create_shaking_effect(sol, m)
                # sol.draw_results("after")
                # sol.print()
                sol = Implementation.vnd(sol, m, rcl_size, seed, tabu_size, prob)

                # sol.store_results(filename)
                sol.store_results("sol.txt")

                name = str(round(sol.obj, 2)) + "_" + str(rcl_size) + "_" + str(seed) + "_tb_" + str(tabu_size) + "_prob" + str(prob)
                sol.draw_results(name)
