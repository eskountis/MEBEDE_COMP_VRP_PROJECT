from Competitional import Starting_Sol, Search
from Competitional.Model import Model

m = Model()
m.build_model()
locations = m.customers
time_matrix = m.time_matrix
depot = m.allNodes[0]

radius = 20

rcl_size = 2
seed = 110
tabu_size = 20
prob = 0.65
sol = Starting_Sol.route_clustering_with_tsp_nearest(radius, m)
print(sol.obj)
sol = Search.tabu_search(sol, m, rcl_size, seed, tabu_size, prob)

sol.store_results("8180023.txt")

name = str(round(sol.obj, 3)) + "_" + str(rcl_size) + "_" + str(seed) + "_tb_" + str(tabu_size) + "_prob" + str(prob)
sol.draw_results(name)
