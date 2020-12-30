import random

from Competitional.model import Route, Solution, Chain


def test_validity_of_solution(sol, time_matrix):
    routes, obj = sol.routes, sol.obj
    eps = 0.01
    costs = []
    for r in routes:
        cost = sum([time_matrix[r.nodes[i].id][r.nodes[i+1].id] for i in range(len(r.nodes) - 1)])
        if abs(cost - r.time) > eps:
            return False
        costs.append(cost)
    true_obj = max(costs)
    if abs(true_obj - obj) < eps:
        return True
    else:
        return False


def apply_worst_fit(sorted_nodes, depot, time_matrix, m):
    routes = [Route(i+1, [depot], 0, 0) for i in range(m.vehicles)]  # nodes in bin with bin's total demand
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
    return routes


def swap_move(sol, time_matrix, m, iterations, rcl_size, seed, k, n):
    optimal = False
    times = 1
    random.seed(seed)
    while not optimal:
        sol = Solution(times, [Route(r.id, [node for node in r.nodes], r.time, r.demand) for r in sol.routes])
        routes = sol.routes
        top_moves = []
        least_cost_difference = 0
        for r in range(len(routes)):  # apply 1-1 exchange operator
            curr_route = routes[r]
            for i in range(1, len(curr_route.nodes) - k):
                curr_pred = curr_route.nodes[i-1]
                chain1 = curr_route.nodes[i:i+k]
                chain1_time = sum([time_matrix[chain1[i].id][chain1[i + 1].id] for i in range(len(chain1) - 1)])
                curr_chain = Chain(chain1, chain1_time, sum([node.demand for node in chain1]), chain1[0], chain1[-1])
                curr_succ = curr_route.nodes[i+k]

                for w in range(r, len(routes)):  # search all the nodes after the curr_chain to find a better solution
                    new_route = routes[w]
                    for j in range(i + k, len(new_route.nodes) - n):
                        new_pred = new_route.nodes[j-1]
                        chain2 = new_route.nodes[j:j+n]
                        chain2_time = sum([time_matrix[chain2[i].id][chain2[i+1].id] for i in range(len(chain2) - 1)])
                        new_chain = Chain(chain2, chain2_time, sum([node.demand for node in chain2]), chain2[0], chain2[-1])
                        new_succ = new_route.nodes[j+n]

                        if curr_route.id != new_route.id:  # check capacity constraints for different routes
                            dem_difference = curr_chain.demand - new_chain.demand
                            if dem_difference > 0 and new_route.demand + dem_difference > m.truck_capacity:
                                continue
                            elif dem_difference < 0 and curr_route.demand + abs(dem_difference) > m.truck_capacity:
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

                        if cost_difference < least_cost_difference:
                            # store the route object, its cost change and the node's index in order to make the swap
                            least_cost_chain1 = [r, i, cost_added1 - cost_reduced1, curr_chain]
                            least_cost_chain2 = [w, j, cost_added2 - cost_reduced2, new_chain]
                            if len(top_moves) < rcl_size:  # fill the list up to the required size
                                top_moves.append([least_cost_chain1, least_cost_chain2])
                                if len(top_moves) == rcl_size:  # when list goes full, set the comparison criteria
                                    move_with_max_cost = max(top_moves, key=lambda x: x[0][2] + x[1][2])
                                    least_cost_difference = move_with_max_cost[0][2] + move_with_max_cost[1][2]
                            else:
                                top_moves.remove(move_with_max_cost)
                                top_moves.append([least_cost_chain1, least_cost_chain2])
                                move_with_max_cost = max(top_moves, key=lambda x: x[0][2] + x[1][2])
                                least_cost_difference = move_with_max_cost[0][2] + move_with_max_cost[1][2]

        if len(top_moves) == 0:
            print("Local Optimal")
            optimal = True
        else:
            selected_move = top_moves[random.randint(0, len(top_moves) - 1)]
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
            insert_in_same_route_error = 0  # correct,if needed, the position in which the second route will be appended
            if route1.id == route2.id:
                insert_in_same_route_error = n - k
            for node in c1.nodes[::-1]:
                route2.nodes.insert(pos_c2 + insert_in_same_route_error, node)
            route1.time += least_cost_chain1[2] + c2.time - c1.time
            route2.time += least_cost_chain2[2] + c1.time - c2.time
            route1.demand += c2.demand - c1.demand
            route2.demand += c1.demand - c2.demand
            # print(least_cost_difference, route1.id, route2.id, [n1.id for n1 in c1.nodes], [n2.id for n2 in c2.nodes], pos_c1, pos_c2)

        times += 1
        if times > iterations:
            print("COULD HAVE DONE MOREEEEEEEEEE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            break

        obj = max(routes, key=lambda x: x.time)
        sol.obj = obj.time
        # sol.print()
        # sol.draw_results()

    return sol
