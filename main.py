import time

from Competitional import implementation
from Competitional.model import Model, Solution

m = Model()
m.build_model()
locations = m.service_locations
time_matrix = m.time_matrix
depot = m.all_nodes[0]


sort_by_demands = sorted(locations, key=lambda x: x.demand, reverse=True)
# construct initial solution by using the worst-fit method to create routes and then solving the tsp problem for each route
routes = implementation.apply_worst_fit(sort_by_demands, depot, time_matrix, m)
initial_sol = Solution(0, routes)

st = time.time()
for rcl_size in range(2, 12):  # determine size of rcl_list used for multi_restart method
    for seed in range(1, 5):
        sol = initial_sol
        for k in range(1, 4)[::-1]:  # implement swaps operator for 3-3, 2-2, and 1-1 node chains
            n = k
            sol = implementation.tabu_search(sol, time_matrix, m, 500, rcl_size, seed, k, n, "")
        sol = implementation.tabu_search(sol, time_matrix, m, 500, rcl_size, seed, 1, 0, "rel")  # implement relocation operator
        sol.print()
        if st > 5 * 60:
            break
        # name = str(round(sol.obj, 2)) + "_" + str(rcl_size) + "_" + str(seed)
        # sol.draw_results(name)
        # filename = "Solutions.txt"
        # sol.store_results(filename)

# filename = "Solutions.txt"
# sols = FileHandler.fetch_solutions(filename, m.all_nodes)
#
# map_by_objective = {}
# for s in sols:
#     if round(s.obj, 2) not in map_by_objective.keys():
#         map_by_objective[round(s.obj, 2)] = [s]
#     else:
#         map_by_objective[round(s.obj, 2)].append(s)
#
# qualified = []
# for i in range(500, 601):
#     rounded_obj = i / 100
#     if rounded_obj in map_by_objective.keys():
#         for s in map_by_objective[rounded_obj]:
#             qualified.append(s)

# rel_1 = 0
# rel_2 = 0
# rel_1_deviations = ["rel_1"]
# rel_2_deviations = ["rel_2"]
# for sol in qualified:
#     rcl_size, seed = sol.rcl_size, sol.seed
#     test_sol = sol
#     for k in range(1, 3)[::-1]:
#         test_sol = uniform_routes.tabu_search(test_sol, time_matrix, m, 500, rcl_size, seed, k, 0, 7, 1, "rel")
#         test_sol.rcl_size, test_sol.seed = rcl_size, seed
#     rel_1_deviations.append(Analytics.routes_st_dev(test_sol))
#     for k in range(1, 2)[::-1]:
#         sol = uniform_routes.tabu_search(sol, time_matrix, m, 500, rcl_size, seed, k, 0, 7, 1, "rel")
#         sol.rcl_size, sol.seed = rcl_size, seed
#     rel_2_deviations.append(Analytics.routes_st_dev(sol))
#
#     if test_sol.obj < sol.obj:
#         sol = test_sol
#         signature = "_rel_2"
#         rel_2 += 1
#     else:
#         rel_1 += 1
#         signature = "_rel_1"
#
#     name = "Relocation_Optimals/" + str(round(sol.obj, 2)) + "_" + str(rcl_size) + "_" + str(seed) + signature
#     sol.draw_results(name)
#     sol.store_results("Relocation_Solutions.txt")
#
# total = rel_1 + rel_2
# print("rel_1:", rel_1, "out of:", total)
# print("rel_2:", rel_2, "out of:", total)
# FileHandler.store_in_csv([rel_1_deviations, rel_2_deviations], "St_Deviations")


# routes = clarke_n_wright.implement(depot, locations, times, m)
# for r in routes:
#     print(r.id, ":", [n.id for n in r.nodes], r.time)
# obj = max(routes, key=lambda x: x.time)
# print("obj:", obj.time)
# m.print_results(routes)

# average_time_costs = [1]
# for n in locations:
#     total = sum([time_matrix[n.id][i.id] + time_matrix[i.id][n.id] for i in m.all_nodes])
#     nums = len(locations) * 2 - 1  # time_matrix[n][0] is not taken into consideration for any n, so reduce nums by 1
#     average_time_costs.append(total/nums)
# sort_by_average_time_cost = sorted(locations, key=lambda x: x.demand, reverse=True)



# qualified = [[7,7], [7,13], [8,10], [4, 19], [9, 17], [10, 17], [3, 12], [4, 10], [4, 16], [5, 15], [5, 16], [6, 10], [6, 14], [7,2],[7, 11],[7,16],[7,18],[8,2],[9,4],[9,7],[10,6]]
# i = 0
# for rcl_size, seed in qualified:
#     routes = uniform_routes.apply_worst_fit(sort_by_demands, depot, time_matrix, m)
#     sol = Solution(0, routes)
#     for k in range(1, 4)[::-1]:
#         n = k
#         sol = uniform_routes.tabu_search(sol, time_matrix, m, 500, rcl_size, seed, k, n, 7, 1, "")
#     sol.rcl_size, sol.seed = rcl_size, seed
#     sol.print()
#     sol.draw_results(str(i))
#     sol.store_results()
#     i += 1
    # for k in range(1, 3)[::-1]:
    #     sol = uniform_routes.tabu_search(sol, time_matrix, m, 500, rcl_size, seed, k, 0, 7, 1, "rel")
    #     # print(sol.id)
    #     sol.draw_results(i)
    #     sol.store_results()
    #     i += 1
    #     sol.print()
    #     print("For relocations of", k, "nodes: obj=", sol.obj)
    #     validity = uniform_routes.test_validity_of_solution(sol, time_matrix)
    #     print("Validity:", validity)
    #     print("Standard deviation of solution:", Analytics.routes_st_dev(sol))
    # print("----------------------------------------")
