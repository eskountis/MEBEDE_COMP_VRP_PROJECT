import time

from Competitional import FileHandler, Implementation
from Competitional.Model import Model, Solution

m = Model()
m.build_model()
locations = m.customers
time_matrix = m.time_matrix
depot = m.allNodes[0]


# filename = "New_Solutions.txt"
# sols_with_obj_up_to = 5.5
# sols = FileHandler.map_by_objective(filename, m.all_nodes, sols_with_obj_up_to)

sort_by_distance_from_depot = sorted(locations, key=lambda x: time_matrix[0][x.id], reverse=True)
sort_by_demands = sorted(locations, key=lambda x: x.demand, reverse=True)
filename = "Good_Solutions.txt"
sorted_nodes = sort_by_distance_from_depot
# signature = "_by_demands"
signature = "_by_distance"

for rcl_size in range(1, 12):  # determine size of rcl_list used for multi_restart method
    for seed in range(1, 21):
        st = time.time()
        routes = Implementation.construct_initial_sol(sorted_nodes, depot, time_matrix, m, rcl_size, seed)
        sol = Implementation.vnd(Solution(0, routes), time_matrix, m, seed, 500, 500, 500)
        end = time.time()
        print((end - st) / 60)

        sol.store_results(filename)
        # sol.print()

        name = str(round(sol.obj, 2)) + "_" + str(rcl_size) + "_" + str(seed) + signature
        sol.draw_results(name)
