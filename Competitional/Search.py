import math
import random
import time

from Competitional import Implementation
from Competitional.Model import Route, Solution, Chain
from Competitional.Solver import TwoOptMove


def calc_cost_difference_for_rel(curr_route, i, k, curr_chain, new_route, j, time_matrix):
    curr_pred = curr_route.nodes[i - 1]
    curr_succ = curr_route.nodes[i + k]
    new_pred = new_route.nodes[j - 1]
    new_succ = new_route.nodes[j]

    cost_added1 = time_matrix[curr_pred.id][curr_succ.id]
    cost_reduced1 = time_matrix[curr_pred.id][curr_chain.first_node.id] + time_matrix[curr_chain.last_node.id][
        curr_succ.id]
    cost_added2 = time_matrix[new_pred.id][curr_chain.first_node.id] + time_matrix[curr_chain.last_node.id][new_succ.id]
    cost_reduced2 = time_matrix[new_pred.id][new_succ.id]

    cost1 = cost_added1 - cost_reduced1
    cost2 = cost_added2 - cost_reduced2
    return cost1, cost2


def calc_cost_difference_for_swaps(curr_route, i, k, curr_chain, new_route, j, new_chain, time_matrix):
    curr_pred = curr_route.nodes[i - 1]
    curr_succ = curr_route.nodes[i + k]
    new_pred = new_route.nodes[j - 1]
    new_succ = new_route.nodes[j + k]

    cost_added1 = time_matrix[curr_pred.id][new_chain.first_node.id] + time_matrix[new_chain.last_node.id][curr_succ.id]
    cost_reduced1 = time_matrix[curr_pred.id][curr_chain.first_node.id] + time_matrix[curr_chain.last_node.id][curr_succ.id]
    cost_added2 = time_matrix[new_pred.id][curr_chain.first_node.id] + time_matrix[curr_chain.last_node.id][new_succ.id]
    cost_reduced2 = time_matrix[new_pred.id][new_chain.first_node.id] + time_matrix[new_chain.last_node.id][new_succ.id]
    if curr_succ.id == new_chain.first_node.id:  # check if the two nodes are adjacent
        # reduce double calculations which occur when the nodes are adjacent
        unnecessary_additions = time_matrix[new_chain.last_node.id][curr_succ.id] + time_matrix[new_pred.id][
            curr_chain.first_node.id]
        cost_added1 += time_matrix[new_chain.last_node.id][curr_chain.first_node.id] - unnecessary_additions
        cost_reduced1 -= time_matrix[curr_chain.last_node.id][new_chain.first_node.id]

    cost1 = cost_added1 - cost_reduced1
    cost2 = cost_added2 - cost_reduced2
    return cost1, cost2


def calc_obj_difference(routes, chain1, chain2):
    r, w = chain1[0], chain2[0]
    curr_chain, new_chain = chain1[3], chain2[3]
    route1_cost_difference = chain1[2] + new_chain.time - curr_chain.time
    route2_cost_difference = chain2[2] + curr_chain.time - new_chain.time
    crucial_time = max(routes, key=lambda x: x.time).time  # the objective
    route_times = [route.time for route in routes]
    route_times[r] += route1_cost_difference
    route_times[w] += route2_cost_difference

    return max(route_times) - crucial_time


def relocation_move(curr_sol, time_matrix, dist_matrix, truck_capacity, rcl_size, k, tabu_list, iterations_unchanged):
    routes = curr_sol.routes
    top_moves = []
    worst_best_cost_reduction = 2 ** 10
    worst_best_obj_reduction = 2 ** 10
    for r in range(len(routes)):  # apply relocation operator
        curr_route = routes[r]
        for i in range(1, len(curr_route.nodes) - k):
            chain1 = curr_route.nodes[i:i + k]
            chain1_time = sum([time_matrix[chain1[i].id][chain1[i + 1].id] for i in range(len(chain1) - 1)])
            curr_chain = Chain(chain1, chain1_time, sum([node.demand for node in chain1]), chain1[0], chain1[-1])

            for w in range(len(routes)):
                new_route = routes[w]
                for j in range(1, len(new_route.nodes)):
                    if r == w and j in range(i, i + k + 1):
                        continue

                    new_pred = new_route.nodes[j - 1]
                    new_succ = new_route.nodes[j]

                    if new_succ.id != 0 or new_pred.id != 0:  # if the route is not empty, check the distances
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

                    route1_cost_difference, route2_cost_difference = calc_cost_difference_for_rel(curr_route, i, k, curr_chain, new_route, j, time_matrix)
                    cost_difference = route1_cost_difference + route2_cost_difference

                    # store the route object, its cost change and the node's index in order to make the swap
                    least_cost_chain1 = [r, i, route1_cost_difference, curr_chain]
                    least_cost_chain2 = [w, j, route2_cost_difference, Chain([], 0, 0, None, None)]
                    # calculate move's impact on the objective function
                    obj_difference = calc_obj_difference(routes, least_cost_chain1, least_cost_chain2)
                    if len(new_route.nodes) == 2:  # if there is an empty route, make the relocation move be applied there
                        obj_difference -= 1

                    if obj_difference < worst_best_obj_reduction or (0 <= worst_best_obj_reduction - obj_difference <= 0.0000000000000001 and cost_difference < worst_best_cost_reduction):
                        move = [r, w, i, j]
                        if move in [mv for tb_mv in tabu_list for mv in tb_mv]:  # check tabu restrictions
                            if obj_difference >= 0 and cost_difference >= 0:  # if move in tabu, check aspiration criteria
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
    if best_obj_difference < 0:  # return the best move found if it improves the objective
        selected_move = top_move
        if len(curr_sol.routes[top_move[1][0]].nodes) == 2:
            selected_move[2] += 1
    else:  # if no move can improve the objective, return randomly oe of the top moves to create shaking effect
        selected_move = top_moves[random.randint(0, len(top_moves) - 1)]
    return selected_move


def k_n_swap_move(curr_sol, time_matrix, dist_matrix, truck_capacity, rcl_size, k, n, tabu_list, iterations_unchanged):
    routes = curr_sol.routes
    top_moves = []
    worst_best_cost_reduction = 2 ** 10
    worst_best_obj_reduction = 2 ** 10
    for r in range(len(routes)):  # apply k-n exchange operator
        curr_route = routes[r]
        for i in range(1, len(curr_route.nodes) - k):
            chain1 = curr_route.nodes[i:i + k]
            chain1_time = sum([time_matrix[chain1[i].id][chain1[i + 1].id] for i in range(len(chain1) - 1)])
            curr_chain = Chain(chain1, chain1_time, sum([node.demand for node in chain1]), chain1[0], chain1[-1])

            for w in range(r, len(routes)):  # search all the nodes after the curr_chain to find a better solution
                new_route = routes[w]
                for j in range(i + k, len(new_route.nodes) - n):
                    chain2 = new_route.nodes[j:j + n]
                    chain2_time = sum([time_matrix[chain2[i].id][chain2[i + 1].id] for i in range(len(chain2) - 1)])
                    new_chain = Chain(chain2, chain2_time, sum([node.demand for node in chain2]), chain2[0], chain2[-1])

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

                    route1_cost_difference, route2_cost_difference = calc_cost_difference_for_swaps(curr_route, i, k, curr_chain, new_route, j, new_chain, time_matrix)
                    cost_difference = route1_cost_difference + route2_cost_difference

                    # store the route object, its cost change and the node's index in order to make the swap
                    least_cost_chain1 = [r, i, route1_cost_difference, curr_chain]
                    least_cost_chain2 = [w, j, route2_cost_difference, new_chain]
                    # calculate move's impact on the objective function
                    obj_difference = calc_obj_difference(routes, least_cost_chain1, least_cost_chain2)

                    if obj_difference < worst_best_obj_reduction or (0 <= worst_best_obj_reduction - obj_difference <= 0.0000000000000001 and cost_difference < worst_best_cost_reduction):
                        move = [r, w, i, j]
                        if move in [mv for tb_mv in tabu_list for mv in tb_mv]:  # check tabu restrictions
                            if obj_difference >= 0 and cost_difference >= 0:  # if move in tabu, check aspiration criteria
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


def find_closest_route_to_node(node, routes, m, ids_to_avoid):
    lowest_average_dist = 10 ** 2
    closest_route = None
    for r in routes:
        if r.id in ids_to_avoid:
            continue
        # find the closest route for the node in examination
        total_dist = sum([m.dist_matrix[node.id][n.id] + m.dist_matrix[n.id][node.id] for n in r.nodes[1:]])
        average_dist = total_dist / (2 * (len(r.nodes) - 2))
        if average_dist < lowest_average_dist:
            lowest_average_dist = average_dist
            closest_route = r

    return closest_route


def create_shaking_effect(sol, m):
    ids_to_avoid = []
    for i in range(1):
        crucial_route = min(sol.routes, key=lambda x: x.time)
        nodes_to_be_transposed = crucial_route.nodes[1:-1]
        replacing_route = Route(crucial_route.id, [m.allNodes[0]], 0, 0)
        for node in nodes_to_be_transposed:
            if m.dist_matrix[0][node.id] <= 30:  # if the node is close to the center append it to the replacing route
                if replacing_route.demand + node.demand <= m.capacity:
                    replacing_route.demand += node.demand
                    replacing_route.time += m.time_matrix[replacing_route.nodes[-1].id][node.id]
                    nodes_to_be_transposed.remove(node)
                    replacing_route.nodes.append(node)

        while len(nodes_to_be_transposed) > 0:  # run for the remaining nodes which are not close to the depot
            node = nodes_to_be_transposed.pop(0)
            ids_to_avoid.append(crucial_route.id)
            for route in sol.routes:
                if route.demand + node.demand > m.capacity:
                    ids_to_avoid.append(route.id)
            closest_route = find_closest_route_to_node(node, sol.routes, m, ids_to_avoid)
            closest_route.demand += node.demand
            closest_route.time += m.time_matrix[closest_route.nodes[-2].id][node.id]
            closest_route.nodes.insert(-1, node)

        ids_to_avoid = [replacing_route.id]

        replacing_route.nodes.append(m.allNodes[0])
        sol.routes.remove(crucial_route)
        sol.routes.insert(replacing_route.id, replacing_route)
        sol.obj = max(sol.routes, key=lambda x: x.time).time


def random_move(sol, m):
    shaked_sol = Solution(0, [Route(r.id, [node for node in r.nodes], r.time, r.demand) for r in sol.routes])
    crucial_time = shaked_sol.obj
    moves = []
    top = TwoOptMove()  # initialize two opt move
    solve = Implementation.prepare_for_2_opt(shaked_sol, m)
    solve.find_random_two_opt_move(top)
    solve.ApplyTwoOptMove(top)
    move = [top.positionOfFirstRoute, top.positionOfSecondRoute, top.positionOfFirstNode, top.positionOfSecondNode]
    shaked_sol = Implementation.convert_solution(1, solve.bestSolution)
    moves.append(move)

    return shaked_sol, moves


def tabu_search(sol, m, rcl_size, seed, tabu_list_size, prob_for_rel):
    tabu_list = []  # list containing all the prohibited moves for a specific number of iterations
    random.seed(seed)
    i = 1
    iterations_unchanged = 0  # count the times the objective is stuck and doesn't change
    times_stuck = 0  # count the times the unstuck of the objective didn't have any effect
    curr_sol = Solution(i, [Route(r.id, [node for node in r.nodes], r.time, r.demand) for r in sol.routes])
    total_best = curr_sol  # store the best objective encountered so far in tabu search
    termination_condition = False
    st = time.time()
    while not termination_condition:
        r = random.random()
        if r < prob_for_rel:  # apply relocation operator
            print("Rel")
            selected_move = relocation_move(curr_sol, m.time_matrix, m.dist_matrix, m.capacity, rcl_size, 1, tabu_list, iterations_unchanged)  # find best move
        elif r < 0.9:  # apply two-opt operator
            print("2opt")
            top = TwoOptMove()  # initialize two opt move
            solve = Implementation.prepare_for_2_opt(curr_sol, m)
            solve.FindBestTwoOptMove(top, tabu_list, rcl_size)
            if top.positionOfFirstRoute is None:
                selected_move = None
        else:  # apply k-n swaps operator
            print("Swap")
            selected_move = k_n_swap_move(curr_sol, m.time_matrix, m.dist_matrix, m.capacity, rcl_size, 1, 1, tabu_list, iterations_unchanged)  # find best move
        if selected_move is None:
            termination_condition = True
            continue

        if r < prob_for_rel:  # apply relocation move
            new_sol, move = apply_relocation_move(i + 1, curr_sol, selected_move, 1)  # apply the move to create a new dummy solution
        elif r < 0.9:  # apply two-opt move
            solve.ApplyTwoOptMove(top)
            move = [top.positionOfFirstRoute, top.positionOfSecondRoute, top.positionOfFirstNode, top.positionOfSecondNode]
            new_sol = Implementation.convert_solution(i + 1, solve.bestSolution)
        else:  # apply k-n swaps
            new_sol, move = apply_k_n_swaps_move(i + 1, curr_sol, selected_move, 1, 1)  # apply the move to create a new dummy solution
        new_sol.rcl_size, new_sol.seed = rcl_size, seed

        if abs(new_sol.obj - curr_sol.obj) < 0.0001:  # check if the solution's objective hasn't changed
            iterations_unchanged += 1
            if iterations_unchanged >= 25:
                print("Stuck - Applying random 2-opt")
                times_stuck += 1
                if times_stuck >= 4:
                    print("Shaking")
                    create_shaking_effect(new_sol, m)
                    times_stuck = 0
                else:
                    iterations_unchanged = 0
                    new_sol, move = random_move(new_sol, m)
        else:
            iterations_unchanged = 0

        print(new_sol.obj)
        if new_sol.obj <= total_best.obj:  # check if the new solution is better than the current best
            if new_sol.obj < total_best.obj:
                times_stuck = 0
            total_best = new_sol

        time_running = (time.time() - st) / 60
        if time_running > 5:
            print(time_running, "minutes")
            termination_condition = True
            continue

        curr_sol = new_sol

        if move is not None:
            tabu_list.append(move)  # update tabu list
        if len(tabu_list) == tabu_list_size:
            tabu_list.pop(random.randint(0, tabu_list_size - 1))  # remove an element of the list when it goes full

    total_best.print()
    # Analytics.visualize_sol_evolvement(np.arange(len(track_of_sols)), track_of_sols)
    return total_best
