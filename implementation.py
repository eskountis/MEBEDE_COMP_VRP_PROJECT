import random

from Competitional.model import Route, Solution, Chain


def test_validity_of_solution(sol, time_matrix):
    routes, obj = sol.routes, sol.obj
    eps = 0.01
    costs = []
    for r in routes:
        cost = sum([time_matrix[r.nodes[i].id][r.nodes[i + 1].id] for i in range(len(r.nodes) - 1)])
        if abs(cost - r.time) > eps:
            return False
        costs.append(cost)
    true_obj = max(costs)
    if abs(true_obj - obj) < eps:
        return True
    else:
        return False


def apply_worst_fit(sorted_nodes, depot, time_matrix, m):
    routes = [Route(i, [depot], 0, 0) for i in range(m.vehicles)]  # nodes in bin with bin's total demand
    available_nodes = [n for n in sorted_nodes]
    while len(available_nodes) > 0:
        node_to_be_added = available_nodes[0]
        available_bins = [b for b in routes if b.demand + node_to_be_added.demand <= m.truck_capacity]
        bin_to_be_added_to = min(available_bins, key=lambda x: x.demand)
        bin_to_be_added_to.nodes.append(node_to_be_added)
        bin_to_be_added_to.time += time_matrix[bin_to_be_added_to.nodes[-2].id][bin_to_be_added_to.nodes[-1].id]
        bin_to_be_added_to.demand += node_to_be_added.demand
        available_nodes.remove(node_to_be_added)
    for r in routes:  # add depot at the end of each route
        r.nodes.append(depot)

    enhanced_routes = []  # apply tsp nearest algorithm in each route to create a better initial solution
    for r in routes:
        enhanced_routes.append(tsp_nearest(r, time_matrix))
    return enhanced_routes

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


def calc_obj_difference(routes, chain1, chain2):
    r, w = chain1[0], chain2[0]
    route1_cost_difference, route2_cost_difference = chain1[2], chain2[2]
    curr_chain, new_chain = chain1[3], chain2[3]
    obj_difference = 0
    crucial_route = max(routes, key=lambda x: x.time)  # the route that determines the objective
    # check if the crucial route's time will be reduced when the move is applied
    if crucial_route.id == r:
        obj_difference += route1_cost_difference
    if crucial_route.id == w:
        obj_difference += route2_cost_difference

    # check if the move applied will cause another route to become crucial
    new_route1_cost_difference = route1_cost_difference + new_chain.time - curr_chain.time
    if routes[r].time + new_route1_cost_difference > crucial_route.time:
        obj_difference = new_route1_cost_difference
    new_route2_cost_difference = route2_cost_difference + curr_chain.time - new_chain.time
    if routes[w].time + new_route2_cost_difference > crucial_route.time:
        obj_difference = new_route2_cost_difference

    return obj_difference


def relocation_move(routes, time_matrix, truck_capacity, rcl_size, k, tabu_list, asp_criteria):
    top_moves = []
    local_best = 2 ** 500
    for r in range(len(routes)):  # apply relocation operator
        curr_route = routes[r]
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

                    if r != w:  # check capacity constraints for different routes
                        if new_route.demand + curr_chain.demand > truck_capacity:
                            continue

                    cost_added1 = time_matrix[curr_pred.id][curr_succ.id]
                    cost_reduced1 = time_matrix[curr_pred.id][curr_chain.first_node.id] + time_matrix[curr_chain.last_node.id][curr_succ.id]
                    cost_added2 = time_matrix[new_pred.id][curr_chain.first_node.id] + time_matrix[curr_chain.last_node.id][new_succ.id]
                    cost_reduced2 = time_matrix[new_pred.id][new_succ.id]
                    cost_difference = cost_added1 + cost_added2 - (cost_reduced1 + cost_reduced2)

                    if cost_difference < local_best:
                        # store the route object, its cost change and the node's index in order to make the swap
                        least_cost_chain1 = [r, i, cost_added1 - cost_reduced1, curr_chain]
                        least_cost_chain2 = [w, j, cost_added2 - cost_reduced2, Chain([], 0, 0, None, None)]

                        move = [r, w, [n1.id for n1 in curr_chain.nodes], i, j]
                        if move in tabu_list:  # check tabu restrictions
                            obj_difference = calc_obj_difference(routes, least_cost_chain1, least_cost_chain2)
                            if obj_difference >= asp_criteria:  # if move in tabu, check aspiration criteria
                                continue

                        if len(top_moves) < rcl_size:  # fill the list up to the required size
                            top_moves.append([least_cost_chain1, least_cost_chain2])
                            if len(top_moves) == rcl_size:  # when list goes full, set the comparison criteria
                                move_with_max_cost = max(top_moves, key=lambda x: x[0][2] + x[1][2])
                                local_best = move_with_max_cost[0][2] + move_with_max_cost[1][2]
                        else:
                            top_moves.remove(move_with_max_cost)
                            top_moves.append([least_cost_chain1, least_cost_chain2])
                            move_with_max_cost = max(top_moves, key=lambda x: x[0][2] + x[1][2])
                            local_best = move_with_max_cost[0][2] + move_with_max_cost[1][2]
    return top_moves


def k_n_swap_move(routes, time_matrix, truck_capacity, rcl_size, k, n, tabu_list, asp_criteria):
    top_moves = []
    local_best = 2 ** 500
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
                    cost_difference = cost_added1 + cost_added2 - (cost_reduced1 + cost_reduced2)

                    if cost_difference < local_best:
                        # store the route object, its cost change and the node's index in order to make the swap
                        least_cost_chain1 = [r, i, cost_added1 - cost_reduced1, curr_chain]
                        least_cost_chain2 = [w, j, cost_added2 - cost_reduced2, new_chain]

                        move = [r, w, [n1.id for n1 in curr_chain.nodes], [n2.id for n2 in new_chain.nodes], i, j]
                        if move in tabu_list:  # check tabu restrictions
                            obj_difference = calc_obj_difference(routes, least_cost_chain1, least_cost_chain2)
                            if obj_difference >= asp_criteria:  # if move in tabu, check aspiration criteria
                                continue

                        if len(top_moves) < rcl_size:  # fill the list up to the required size
                            top_moves.append([least_cost_chain1, least_cost_chain2])
                            if len(top_moves) == rcl_size:  # when list goes full, set the comparison criteria
                                move_with_max_cost = max(top_moves, key=lambda x: x[0][2] + x[1][2])
                                local_best = move_with_max_cost[0][2] + move_with_max_cost[1][2]
                        else:
                            top_moves.remove(move_with_max_cost)
                            top_moves.append([least_cost_chain1, least_cost_chain2])
                            move_with_max_cost = max(top_moves, key=lambda x: x[0][2] + x[1][2])
                            local_best = move_with_max_cost[0][2] + move_with_max_cost[1][2]
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
    move = [route2.id, route1.id, [n1.id for n1 in c1.nodes], pos_c2, pos_c1]  # store the inverse move into tabu list
    dummy_sol.obj = max(dummy_sol.routes, key=lambda x: x.time).time

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
    move = [route1.id, route2.id, [n2.id for n2 in c2.nodes], [n1.id for n1 in c1.nodes], pos_c1, pos_c2]  # store the inverse move into tabu list
    dummy_sol.obj = max(dummy_sol.routes, key=lambda x: x.time).time

    return dummy_sol, move


def tabu_search(sol, time_matrix, m, iterations, rcl_size, seed, k, n, switch):
    tabu_list = []  # list containing all the prohibited moves for a specific number of iterations
    tabu_list_size = 7  # the number of iterations for which a tabu element is forbidden from being accepted
    mutation_probability = 1  # probability that a new solution created is assigned to curr_sol for the next iteration
    random.seed(seed)
    i = 1
    curr_sol = Solution(i, [Route(r.id, [node for node in r.nodes], r.time, r.demand) for r in sol.routes])
    total_best = curr_sol  # store the best objective encountered so far in tabu search
    termination_condition = False
    while not termination_condition:
        asp_criteria = round(total_best.obj - curr_sol.obj, 2)  # set aspiration criteria for override of a tabu move
        if switch == "rel":  # apply relocation operator
            top_moves = relocation_move(curr_sol.routes, time_matrix, m.truck_capacity, rcl_size, k, tabu_list, asp_criteria)  # find best move
        else:  # apply k-n swaps operator
            top_moves = k_n_swap_move(curr_sol.routes, time_matrix, m.truck_capacity, rcl_size, k, n, tabu_list, asp_criteria)  # find best move
        if len(top_moves) == 0:  # check if all neighbors are in tabu list
            termination_condition = True
            continue

        selected_move = top_moves[random.randint(0, len(top_moves) - 1)]
        if switch == "rel":  # apply relocation move
            new_sol, move = apply_relocation_move(i + 1, curr_sol, selected_move, k)  # apply the move to create a new dummy solution
        else:  # apply k-n swaps
            new_sol, move = apply_k_n_swaps_move(i + 1, curr_sol, selected_move, k, n)  # apply the move to create a new dummy solution

        if new_sol.obj <= total_best.obj:  # check if the new solution is better than the current best
            total_best = new_sol

        if random.random() <= mutation_probability:  # mutate the current solution according to the given probability
            curr_sol = new_sol

        tabu_list.append(move)  # update tabu list
        if len(tabu_list) == tabu_list_size:
            tabu_list.pop(0)  # remove the first element of the list when it goes full

        i += 1
        if i > iterations:
            termination_condition = True

    return total_best
