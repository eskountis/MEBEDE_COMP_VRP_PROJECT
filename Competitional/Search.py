import random
import numpy as np

from Competitional import Analytics, Solver, Implementation
from Competitional.Model import Route, Solution, Chain
from Competitional.Solver import TwoOptMove


def test_validity_of_solution(sol, time_matrix):
    routes, obj = sol.routes, sol.obj
    eps = 0.01
    costs = []
    for r in routes:
        cost = r.calc_route_cost(time_matrix)
        if abs(cost - r.time) > eps:
            return False
        costs.append(cost)
    true_obj = max(costs)
    print(true_obj, obj)
    if abs(true_obj - obj) < eps:
        return True
    else:
        return False


def apply_worst_fit(sorted_nodes, depot, time_matrix, m, rcl_size, seed):
    random.seed(seed)
    routes = [Route(i, [depot], 0, 0) for i in range(m.vehicles)]  # nodes in bin with bin's total demand
    available_nodes = [n for n in sorted_nodes]
    while len(available_nodes) > 0:
        chosen_node_index = random.randint(0, min(rcl_size, len(available_nodes) - 1))  # apply multi-restart method
        node_to_be_added = available_nodes[chosen_node_index]
        available_bins = [b for b in routes if b.demand + node_to_be_added.demand <= m.capacity]
        bin_to_be_added_to = min(available_bins, key=lambda x: x.demand)
        bin_to_be_added_to.nodes.append(node_to_be_added)
        bin_to_be_added_to.time += time_matrix[bin_to_be_added_to.nodes[-2].id][bin_to_be_added_to.nodes[-1].id]
        bin_to_be_added_to.demand += node_to_be_added.demand
        available_nodes.remove(node_to_be_added)
    for r in routes:  # add depot at the end of each route
        r.nodes.append(depot)

    return routes


def tsp_nearest(route, time_matrix):
    depot = route.nodes[0]
    available_nodes = [n for n in route.nodes[1:-1]]
    opt_route = Route(route.id, [depot, depot], 0, route.demand)
    while len(available_nodes) > 0:
        min_cost_addition = 500
        min_cost_node = None
        min_cost_pos = -1
        for pos in range(1, len(opt_route.nodes)):
            pred = opt_route.nodes[pos - 1]
            succ = opt_route.nodes[pos]
            best_node = min(available_nodes, key=lambda x: time_matrix[pred.id][x.id] + time_matrix[x.id][succ.id] - time_matrix[pred.id][succ.id])
            best_cost_addition = time_matrix[pred.id][best_node.id] + time_matrix[best_node.id][succ.id] - time_matrix[pred.id][succ.id]
            if best_cost_addition < min_cost_addition:
                min_cost_addition = best_cost_addition
                min_cost_node = best_node
                min_cost_pos = pos
        opt_route.nodes.insert(min_cost_pos, min_cost_node)
        opt_route.time += min_cost_addition
        available_nodes.remove(min_cost_node)
    return opt_route


def exclude_outliers(sol):
    sol.update_median_and_st_dev()
    normal_values = []
    for r in sol.routes:
        if sol.median - 1 * sol.st_dev < r.time:
            normal_values.append(r)
        else:
            # instead of a route with small cost, insert an empty route object so that this route isn't checked
            normal_values.append(Route(0, [], 0, 0))
    return normal_values


def calc_obj_difference(routes, chain1, chain2):
    r, w = chain1[0], chain2[0]
    curr_chain, new_chain = chain1[3], chain2[3]
    route1_cost_difference = chain1[2] + new_chain.time - curr_chain.time
    route2_cost_difference = chain2[2] + curr_chain.time - new_chain.time
    crucial_time = max(routes, key=lambda x: x.time).time  # the objective
    # crucial_routes = list(filter(lambda x: abs(x.time - crucial_time) < 0.005, routes))  # the routes that determine the objective
    # obj_difference = 0
    # if (r == w and len(crucial_routes) > 1) or (r != w and len(crucial_routes) > 2):
    #     return obj_difference  # obj will not change because there are two many crucial routes
    # else:
    route_times = [route.time for route in routes]
    route_times[r] += route1_cost_difference
    route_times[w] += route2_cost_difference
    return max(route_times) - crucial_time
        # check if the crucial routes' time will be reduced when the move is applied
    #     if len(crucial_routes) == 1:
    #
    # # print(crucial_route.id, r, w)
    # # print(crucial_route.time, route1_cost_difference, route2_cost_difference)
    # if crucial_route.id == r:
    #     obj_difference += route1_cost_difference
    #     crucial_time += route1_cost_difference
    # elif crucial_route.id == w:
    #     obj_difference += route2_cost_difference
    #     crucial_time += route2_cost_difference
    # else:
    #     obj_difference = 0
    #
    # new_route1_time = routes[r].time + route1_cost_difference
    # new_route2_time = routes[w].time + route2_cost_difference
    # # check if the move applied will cause another route to become crucial
    # if new_route1_time > crucial_time:
    #     obj_difference = new_route1_time - crucial_route.time
    #     crucial_time = new_route1_time  # update crucial_time to check if the other route has been increased furthermore
    # if new_route2_time > crucial_time:
    #     obj_difference = new_route2_time - crucial_route.time
    #
    # return obj_difference


def relocation_move(curr_sol, time_matrix, dist_matrix, truck_capacity, rcl_size, k, tabu_list, asp_criteria, iterations_unchanged):
    routes = curr_sol.routes
    top_moves = []
    worst_best_cost_reduction = 2 ** 10
    worst_best_obj_reduction = 2 ** 10
    normal_routes = exclude_outliers(curr_sol)  # take into account only the routes with average or high time cost
    for r in range(len(normal_routes)):  # apply relocation operator
        curr_route = normal_routes[r]
        for i in range(1, len(curr_route.nodes) - k):
            curr_pred = curr_route.nodes[i - 1]
            chain1 = curr_route.nodes[i:i + k]
            chain1_time = sum([time_matrix[chain1[i].id][chain1[i + 1].id] for i in range(len(chain1) - 1)])
            curr_chain = Chain(chain1, chain1_time, sum([node.demand for node in chain1]), chain1[0], chain1[-1])
            curr_succ = curr_route.nodes[i + k]

            for w in range(len(routes)):
                new_route = routes[w]
                for j in range(1, len(new_route.nodes)):
                    if r == w and j in range(i, i + k + 1):
                        continue

                    new_pred = new_route.nodes[j - 1]
                    new_succ = new_route.nodes[j]

                    # check if the all the nodes at the relocation move examined are out of scope
                    in_scope = True
                    for n in curr_chain.nodes:
                        if dist_matrix[n.id][new_pred.id] > n.scope_radius and dist_matrix[n.id][new_succ.id] > n.scope_radius:
                            in_scope = False
                            break
                    if not in_scope:
                        continue

                    if r != w:  # check capacity constraints for different routes
                        if new_route.demand + curr_chain.demand > truck_capacity:
                            continue

                    cost_added1 = time_matrix[curr_pred.id][curr_succ.id]
                    cost_reduced1 = time_matrix[curr_pred.id][curr_chain.first_node.id] + time_matrix[curr_chain.last_node.id][curr_succ.id]
                    cost_added2 = time_matrix[new_pred.id][curr_chain.first_node.id] + time_matrix[curr_chain.last_node.id][new_succ.id]
                    cost_reduced2 = time_matrix[new_pred.id][new_succ.id]
                    cost_difference = cost_added1 - cost_reduced1 + cost_added2 - cost_reduced2

                    # store the route object, its cost change and the node's index in order to make the swap
                    least_cost_chain1 = [r, i, cost_added1 - cost_reduced1, curr_chain]
                    least_cost_chain2 = [w, j, cost_added2 - cost_reduced2, Chain([], 0, 0, None, None)]
                    # calculate move's impact on the objective function
                    obj_difference = calc_obj_difference(routes, least_cost_chain1, least_cost_chain2)

                    if obj_difference < worst_best_obj_reduction or (0 <= worst_best_obj_reduction - obj_difference <= 0.0000000000000001 and cost_difference < worst_best_cost_reduction):
                        move = [r, w, i, j]
                        if move in [mv for tb_mv in tabu_list for mv in tb_mv]:  # check tabu restrictions
                            if obj_difference >= asp_criteria:  # if move in tabu, check aspiration criteria
                                continue

                        if len(top_moves) < rcl_size:  # fill the list up to the required size
                            top_moves.append([least_cost_chain1, least_cost_chain2, obj_difference])
                            if len(top_moves) == rcl_size:  # when list goes full, set the comparison criteria
                                worst_obj_difference = max(top_moves, key=lambda x: x[2])[2]
                                worst_best_move = max(filter(lambda x: abs(x[2] - worst_obj_difference) < 0.001, top_moves), key=lambda x: x[0][2] + x[1][2])
                                worst_best_cost_reduction = worst_best_move[0][2] + worst_best_move[1][2]
                                worst_best_obj_reduction = worst_best_move[2]
                        else:
                            top_moves.remove(worst_best_move)
                            top_moves.append([least_cost_chain1, least_cost_chain2, obj_difference])
                            worst_obj_difference = max(top_moves, key=lambda x: x[2])[2]
                            worst_best_move = max(filter(lambda x: abs(x[2] - worst_obj_difference) < 0.001, top_moves), key=lambda x: x[0][2] + x[1][2])
                            worst_best_cost_reduction = worst_best_move[0][2] + worst_best_move[1][2]
                            worst_best_obj_reduction = worst_best_move[2]

    top_move = min(top_moves, key=lambda x: x[2])
    best_obj_difference = top_move[2]
    if best_obj_difference < 0 or (iterations_unchanged < 10):  # return the best move found if it improves the objective
        selected_move = top_move
    else:  # if no move can improve the objective, return randomly oe of the top moves to create shaking effect
        selected_move = top_moves[random.randint(0, len(top_moves) - 1)]
    return selected_move

    return top_moves


def k_n_swap_move(routes, time_matrix, dist_matrix, truck_capacity, rcl_size, k, n, tabu_list, asp_criteria, iterations_unchanged):
    top_moves = []
    worst_best_cost_reduction = 2 ** 10
    worst_best_obj_reduction = 2 ** 10
    for r in range(len(routes)):  # apply k-n exchange operator
        curr_route = routes[r]
        for i in range(1, len(curr_route.nodes) - k):
            curr_pred = curr_route.nodes[i - 1]
            chain1 = curr_route.nodes[i:i + k]
            chain1_time = sum([time_matrix[chain1[i].id][chain1[i + 1].id] for i in range(len(chain1) - 1)])
            curr_chain = Chain(chain1, chain1_time, sum([node.demand for node in chain1]), chain1[0], chain1[-1])
            curr_succ = curr_route.nodes[i + k]

            for w in range(r, len(routes)):  # search all the nodes after the curr_chain to find a better solution
                new_route = routes[w]
                for j in range(i + k, len(new_route.nodes) - n):
                    new_pred = new_route.nodes[j - 1]
                    chain2 = new_route.nodes[j:j + n]
                    chain2_time = sum([time_matrix[chain2[i].id][chain2[i + 1].id] for i in range(len(chain2) - 1)])
                    new_chain = Chain(chain2, chain2_time, sum([node.demand for node in chain2]), chain2[0], chain2[-1])
                    new_succ = new_route.nodes[j + n]

                    # check if the all the nodes at the swap move examined are out of scope
                    in_scope = False
                    for n1 in curr_chain.nodes:
                        for n2 in new_chain.nodes:
                            if dist_matrix[n1.id][n2.id] <= n1.scope_radius:
                                in_scope = True
                                break
                        if in_scope:
                            break
                    if not in_scope:
                        continue

                    if r != w:  # check capacity constraints for different routes
                        dem_difference = curr_chain.demand - new_chain.demand
                        if dem_difference > 0 and new_route.demand + dem_difference > truck_capacity:
                            continue
                        elif dem_difference < 0 and curr_route.demand + abs(dem_difference) > truck_capacity:
                            continue

                    cost_added1 = time_matrix[curr_pred.id][new_chain.first_node.id] + time_matrix[new_chain.last_node.id][curr_succ.id]
                    cost_reduced1 = time_matrix[curr_pred.id][curr_chain.first_node.id] + time_matrix[curr_chain.last_node.id][curr_succ.id]
                    cost_added2 = time_matrix[new_pred.id][curr_chain.first_node.id] + time_matrix[curr_chain.last_node.id][new_succ.id]
                    cost_reduced2 = time_matrix[new_pred.id][new_chain.first_node.id] + time_matrix[new_chain.last_node.id][new_succ.id]
                    if curr_succ.id == new_chain.first_node.id:  # check if the two nodes are adjacent
                        # reduce double calculations which occur when the nodes are adjacent
                        unnecessary_additions = time_matrix[new_chain.last_node.id][curr_succ.id] + time_matrix[new_pred.id][curr_chain.first_node.id]
                        cost_added1 += time_matrix[new_chain.last_node.id][curr_chain.first_node.id] - unnecessary_additions
                        cost_reduced1 -= time_matrix[curr_chain.last_node.id][new_chain.first_node.id]
                    cost_difference = cost_added1 - cost_reduced1 + cost_added2 - cost_reduced2

                    # store the route object, its cost change and the node's index in order to make the swap
                    least_cost_chain1 = [r, i, cost_added1 - cost_reduced1, curr_chain]
                    least_cost_chain2 = [w, j, cost_added2 - cost_reduced2, new_chain]
                    # calculate move's impact on the objective function
                    obj_difference = calc_obj_difference(routes, least_cost_chain1, least_cost_chain2)

                    if obj_difference < worst_best_obj_reduction or (0 <= worst_best_obj_reduction - obj_difference <= 0.0000000000000001 and cost_difference < worst_best_cost_reduction):
                        move = [r, w, i, j]
                        if move in tabu_list:  # check tabu restrictions
                            if obj_difference >= asp_criteria:  # if move in tabu, check aspiration criteria
                                continue

                        if len(top_moves) < rcl_size:  # fill the list up to the required size
                            top_moves.append([least_cost_chain1, least_cost_chain2, obj_difference])
                            if len(top_moves) == rcl_size:  # when list goes full, set the comparison criteria
                                worst_obj_difference = max(top_moves, key=lambda x: x[2])[2]
                                worst_best_move = max(filter(lambda x: abs(x[2] - worst_obj_difference) < 0.01, top_moves), key=lambda x: x[0][2] + x[1][2])
                                worst_best_cost_reduction = worst_best_move[0][2] + worst_best_move[1][2]
                                worst_best_obj_reduction = worst_best_move[2]
                        else:
                            top_moves.remove(worst_best_move)
                            top_moves.append([least_cost_chain1, least_cost_chain2, obj_difference])
                            worst_obj_difference = max(top_moves, key=lambda x: x[2])[2]
                            worst_best_move = max(filter(lambda x: abs(x[2] - worst_obj_difference) < 0.01, top_moves), key=lambda x: x[0][2] + x[1][2])
                            worst_best_cost_reduction = worst_best_move[0][2] + worst_best_move[1][2]
                            worst_best_obj_reduction = worst_best_move[2]

    top_move = min(top_moves, key=lambda x: x[2])
    best_obj_difference = top_move[2]
    if best_obj_difference < 0 or (iterations_unchanged < 10):  # return the best move found if it improves the objective
        selected_move = top_move
    else:  # if no move can improve the objective, return randomly one of the top moves to create shaking effect
        selected_move = top_moves[random.randint(0, len(top_moves) - 1)]
    return selected_move

    return top_moves


def apply_relocation_move(i, sol, selected_move, k):
    # copy current solution into a dummy one, so that the previous solution structure is not altered
    dummy_sol = Solution(i, [Route(r.id, [node for node in r.nodes], r.time, r.demand) for r in sol.routes])
    routes = dummy_sol.routes
    least_cost_chain1, least_cost_chain2 = selected_move[0], selected_move[1]
    route1, route2 = routes[least_cost_chain1[0]], routes[least_cost_chain2[0]]
    pos_c1, pos_c2 = least_cost_chain1[1], least_cost_chain2[1]
    c1 = least_cost_chain1[3]
    for i in range(k):
        route1.nodes.pop(pos_c1)
    insert_in_same_route_error = 0  # correct, if needed, the position in which the second route will be appended
    if route1.id == route2.id and pos_c1 < pos_c2:
        insert_in_same_route_error -= k
    for node in c1.nodes[::-1]:
        route2.nodes.insert(pos_c2 + insert_in_same_route_error, node)
    # update routes' time and demand
    route1.time += least_cost_chain1[2] - c1.time
    route2.time += least_cost_chain2[2] + c1.time
    route1.demand -= c1.demand
    route2.demand += c1.demand
    move = [[route2.id, route1.id, pos_c2, pos_c1]]  # store the inverse move into tabu list
    if route1.id == route2.id: # and abs(pos_c1 - pos_c2) == 1:  # if the relocation happens between to adjacent nodes, another move must be added to the tabu
        move.append([route1.id, route2.id, pos_c1, pos_c2])
    dummy_sol.obj = sol.obj + selected_move[2]  # add the objective difference to the current solution

    return dummy_sol, move


def apply_k_n_swaps_move(i, sol, selected_move, k, n):
    # copy current solution into a dummy one, so that the previous solution structure is not altered
    dummy_sol = Solution(i, [Route(r.id, [node for node in r.nodes], r.time, r.demand) for r in sol.routes])
    routes = dummy_sol.routes
    least_cost_chain1, least_cost_chain2 = selected_move[0], selected_move[1]
    route1, route2 = routes[least_cost_chain1[0]], routes[least_cost_chain2[0]]
    pos_c1, pos_c2 = least_cost_chain1[1], least_cost_chain2[1]
    c1, c2 = least_cost_chain1[3], least_cost_chain2[3]
    for i in range(n):  # first pop route2 nodes in case the two chains are adjacent
        route2.nodes.pop(pos_c2)
    for i in range(k):
        route1.nodes.pop(pos_c1)
    for node in c2.nodes[::-1]:
        route1.nodes.insert(pos_c1, node)
    insert_in_same_route_error = 0  # correct, if needed, the position in which the second route will be appended
    if route1.id == route2.id:
        insert_in_same_route_error = n - k
    for node in c1.nodes[::-1]:
        route2.nodes.insert(pos_c2 + insert_in_same_route_error, node)
    # update routes' time and demand
    route1.time += least_cost_chain1[2] + c2.time - c1.time
    route2.time += least_cost_chain2[2] + c1.time - c2.time
    route1.demand += c2.demand - c1.demand
    route2.demand += c1.demand - c2.demand
    move = [route1.id, route2.id, pos_c1, pos_c2]  # store the inverse move into tabu list
    dummy_sol.obj = sol.obj + selected_move[2]  # add the objective difference to the current solution

    return dummy_sol, move


def create_shaking_effect(sol, time_matrix, switch):
    shaked_sol = Solution(0, [Route(r.id, [node for node in r.nodes], r.time, r.demand) for r in sol.routes])
    crucial_route = max(shaked_sol.routes, key=lambda x: x.time)
    if switch == "rel":  # apply a random swap move to the route with the highest cost to shake the relocation operator
        swap_pos1 = random.randint(1, len(crucial_route.nodes) - 2)
        remaining_pos = [i for i in np.arange(1, swap_pos1)] + [j for j in np.arange(swap_pos1 + 1, len(crucial_route.nodes) - 1)]
        swap_pos2 = random.choice(remaining_pos)
        if swap_pos1 > swap_pos2:
            swap_pos1, swap_pos2 = swap_pos2, swap_pos1
        pos2_node = crucial_route.nodes.pop(swap_pos2)
        pos1_node = crucial_route.nodes.pop(swap_pos1)
        crucial_route.nodes.insert(swap_pos1, pos2_node)
        crucial_route.nodes.insert(swap_pos2, pos1_node)
        move = [[crucial_route.id, crucial_route.id, swap_pos1, swap_pos2]]
    else:  # apply a random relocation move to the route with the highest cost to shake the swap operator
        # for i in range(1, len(crucial_route.nodes) - 2):
        #     for j in range(1, len(crucial_route.nodes) - 2):
        remove_pos = random.randint(1, len(crucial_route.nodes) - 2)
        remaining_pos = [i for i in np.arange(1, remove_pos)] + [j for j in np.arange(remove_pos + 1, len(crucial_route.nodes) - 1)]
        insert_pos = random.choice(remaining_pos)
        random_node = crucial_route.nodes.pop(remove_pos)
        crucial_route.nodes.insert(insert_pos, random_node)
        move = [[crucial_route.id, crucial_route.id, insert_pos, remove_pos]]

    crucial_route.time = crucial_route.calc_route_cost(time_matrix)
    shaked_sol.obj = max(shaked_sol.routes, key=lambda x: x.time).time

    return shaked_sol, move


def tabu_search(sol, m, iterations, rcl_size, seed, k, n, switch):
    if switch == "2opt":
        top = TwoOptMove()  # initialize two opt move
        solve = Implementation.prepare_for_2_opt(sol, m)
    track_of_sols = [sol.obj]
    tabu_list = []  # list containing all the prohibited moves for a specific number of iterations
    tabu_list_size = 20  # the number of iterations for which a tabu element is forbidden from being accepted
    mutation_probability = 1  # probability that a new solution created is assigned to curr_sol for the next iteration
    random.seed(seed)
    i = 1
    iterations_unchanged = 0  # count the times the objective is stuck and doesn't change
    times_stuck = 0  # count the times the unstuck of the objective didn't have any effect
    curr_sol = Solution(i, [Route(r.id, [node for node in r.nodes], r.time, r.demand) for r in sol.routes])
    total_best = curr_sol  # store the best objective encountered so far in tabu search
    termination_condition = False
    selected_move = 0
    while not termination_condition:
        asp_criteria = round(total_best.obj - curr_sol.obj, 2)  # set aspiration criteria for override of a tabu move
        if switch == "rel":  # apply relocation operator
            selected_move = relocation_move(curr_sol, m.time_matrix, m.dist_matrix, m.capacity, rcl_size, k, tabu_list, asp_criteria, iterations_unchanged)  # find best move
        elif switch == "2opt":  # apply two-opt operator
            solve.FindBestTwoOptMove(top, tabu_list)
            if top.positionOfFirstRoute is None:
                selected_move = None
        else:  # apply k-n swaps operator
            selected_move = k_n_swap_move(curr_sol.routes, m.time_matrix, m.dist_matrix, m.capacity, rcl_size, k, n, tabu_list, asp_criteria, iterations_unchanged)  # find best move
        if selected_move is None:
            termination_condition = True
            continue

        if switch == "rel":  # apply relocation move
            new_sol, move = apply_relocation_move(i + 1, curr_sol, selected_move, k)  # apply the move to create a new dummy solution
        elif switch == "2opt":  # apply two-opt move
            solve.ApplyTwoOptMove(top)
            move = [top.positionOfFirstRoute, top.positionOfSecondRoute, top.positionOfFirstNode, top.positionOfSecondNode]
            new_sol = Implementation.convert_solution(i + 1, solve.bestSolution)
        else:  # apply k-n swaps
            new_sol, move = apply_k_n_swaps_move(i + 1, curr_sol, selected_move, k, n)  # apply the move to create a new dummy solution

        if abs(new_sol.obj - curr_sol.obj) < 0.0001:  # check if the solution's objective hasn't changed
            iterations_unchanged += 1
            if iterations_unchanged > 10:
                print("Stuck")
                times_stuck += 1
                iterations_unchanged = 0
            if times_stuck >= 3:
                print("Shaking")
                new_sol, move = create_shaking_effect(new_sol, m.time_matrix, switch)
                times_stuck = 0

        print(new_sol.obj)
        if new_sol.obj <= total_best.obj:  # check if the new solution is better than the current best
            if new_sol.obj < total_best.obj:
                iterations_unchanged = 0
                times_stuck = 0
            total_best = new_sol

        if random.random() <= mutation_probability:  # mutate the current solution according to the given probability
            curr_sol = new_sol
            track_of_sols.append(new_sol.obj)

        tabu_list.append(move)  # update tabu list
        if len(tabu_list) == tabu_list_size:
            tabu_list.pop(random.randint(0, tabu_list_size - 1))  # remove an element of the list when it goes full

        i += 1
        if i > iterations:
            termination_condition = True

    # total_best.rcl_size, total_best.seed = rcl_size, seed
    total_best.print()
    validity = test_validity_of_solution(total_best, m.time_matrix)
    print("Validity:", validity)
    Analytics.visualize_sol_evolvement(track_of_sols)
    return total_best
