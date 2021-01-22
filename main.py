from Competitional import Starting_Sol, Search
from Competitional.Model import Model

m = Model()
m.build_model()
locations = m.customers
time_matrix = m.time_matrix
depot = m.allNodes[0]

radius = 20

points = [[2, 110, 20, 0.65], [5, 110, 20, 0.6], [2, 109, 20, 0.65], [7, 102, 20, 0.55]]
for p in points:
    rcl_size, seed, tabu_size, prob = p[0], p[1], 16, p[3]
    sol = Starting_Sol.route_clustering_with_tsp_nearest(radius, m)
    print(sol.obj)
    sol = Search.tabu_search(sol, m, rcl_size, seed, tabu_size, prob)

    sol.store_results("sol.txt")

    name = str(round(sol.obj, 3)) + "_" + str(rcl_size) + "_" + str(seed) + "_tb_" + str(tabu_size) + "_prob" + str(prob)
    sol.draw_results(name)
