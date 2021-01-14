from Competitional import Search, VRP_Model
from Competitional.Model import Route, Solution
from Competitional import Solver


def vnd(initial_sol, time_matrix, m, seed, iterations_for_swap, iterations_for_2opt, iterations_for_rel):
    eps = 0.01
    sol = initial_sol
    # implement swaps operator for 1-1 node chains
    sol = Search.tabu_search(sol, m, iterations_for_swap, 10, seed, 1, 1, "")

    # implement 2opt operator
    sol = Search.tabu_search(sol, m, iterations_for_2opt, 10, seed, 1, 0, "2opt")

    # implement relocation operator
    sol = Search.tabu_search(sol, m, iterations_for_rel, 10, seed, 1, 0, "rel")
    sol.rcl_size, sol.seed = 10, seed
    validity = Search.test_validity_of_solution(sol, time_matrix)
    print("Validity:", validity)

    sol.print()
    if abs(sol.obj - initial_sol.obj) > eps:  # apply vnd method
        sol = vnd(sol, time_matrix, m, seed, iterations_for_swap, iterations_for_2opt, iterations_for_rel)

    return sol


def construct_initial_sol(sorted_nodes, depot, time_matrix, m, rcl_size, seed):
    # construct initial solution by using the worst-fit method to create routes and then solving the tsp problem for each route
    routes = Search.apply_worst_fit(sorted_nodes, depot, time_matrix, m, rcl_size, seed)
    enhanced_routes = []  # apply tsp nearest algorithm in each route to create a better initial solution
    for r in routes:
        enhanced_routes.append(Search.tsp_nearest(r, time_matrix))

    return enhanced_routes


def prepare_for_2_opt(swap_sol, m):
    solve = Solver.Solver(m)

    copy_routes = [VRP_Model.Route(m.allNodes[0], m.capacity) for i in range(m.vehicles)]
    for i in range(m.vehicles):
        copy_routes[i].sequenceOfNodes.pop(-1)
        for n in swap_sol.routes[i].nodes[1:]:
            copy_routes[i].sequenceOfNodes.append(n)
        copy_routes[i].cost = swap_sol.routes[i].time
        copy_routes[i].load = swap_sol.routes[i].demand

    copy_solution = Solver.Solution()
    copy_solution.cost = swap_sol.obj
    copy_solution.routes = copy_routes

    solve.sol = copy_solution
    solve.bestSolution = copy_solution
    solve.overallBestSol = copy_solution

    return solve


def convert_solution(iteration, solution):
    routes = solution.routes
    converted_routes = []
    for i in range(len(routes)):
        nodes = [n for n in routes[i].sequenceOfNodes]
        converted_routes.append(Route(i, nodes, routes[i].cost, routes[i].load))
    return Solution(iteration, converted_routes)



