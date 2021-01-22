from Competitional import VRP_Model
from Competitional.Model import Route, Solution
from Competitional import Solver


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


def test_solution(sol, time_matrix, rt1, rt2):
    routes, obj = sol.routes, sol.cost
    eps = 0.01
    costs = []
    for i in range(len(routes)):
        r = routes[i]
        cost = sum([time_matrix[r.sequenceOfNodes[i].id][r.sequenceOfNodes[i + 1].id] for i in range(len(r.sequenceOfNodes) - 1)])
        if i == rt1 or i == rt2:
            print(cost, r.cost)
        if abs(cost - r.cost) > eps:
            return False
        costs.append(cost)
    true_obj = max(costs)
    print(true_obj, obj)
    if abs(true_obj - obj) < eps:
        return True
    else:
        return False


def test_validity_of_solution(sol, time_matrix):
    routes, obj = sol.routes, sol.obj
    eps = 0.01
    costs = []
    for r in routes:
        cost = r.calc_route_cost(time_matrix)
        # print(cost, r.time)
        if abs(cost - r.time) > eps:
            return False
        costs.append(cost)
    true_obj = max(costs)
    # print(true_obj, obj)
    if abs(true_obj - obj) < eps:
        return True
    else:
        return False
