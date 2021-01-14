from Competitional.Model import Node, Route


def calc_additional_cost(n1: Node, n2: Node, interfering_node: Node, time_matrix):
    return round(time_matrix[n1.id][interfering_node.id] + time_matrix[interfering_node.id][n2.id] - time_matrix[n1.id][n2.id], 2)


def check_for_less_time(i, curr_route, total_demand, available_nodes, curr_best_time, time_matrix, truck_capacity):
    n_times = {n: calc_additional_cost(curr_route[i - 1], curr_route[i], n, time_matrix) for n in available_nodes if total_demand + n.demand <= truck_capacity}
    min_time_n = min(n_times, key=lambda x: n_times[x])
    if n_times[min_time_n] < curr_best_time:
        return min_time_n, n_times[min_time_n]
    else:
        return None


def implement(depot, service_locations, time_matrix, m):
    # dict representing each node's [0, n, 0] route and showing its total time
    single_routes_time = {n: time_matrix[depot.id][n.id] for n in service_locations}
    # all the nodes adjusted in ascending order regarding their route time
    available_nodes = sorted(single_routes_time, key=lambda x: single_routes_time[x])
    first_node = available_nodes.pop(0)
    routes = [Route(1, [depot, first_node, depot], single_routes_time[first_node], first_node.demand)]  # list of route objects
    while len(available_nodes) > 0:
        node_added = available_nodes[0]  # node that is added to the solution and must be removed from the available
        curr_best_time = single_routes_time[node_added]
        min_demand = min(n.demand for n in available_nodes)
        pos = len(routes) if len(routes) < m.vehicles else len(routes) - 1
        route_order = 0
        for route in routes:
            if min_demand + route.demand > m.capacity:  # check if the current route can fit any additional nodes
                continue
            for i in range(1, len(route.nodes)):
                best_opt = check_for_less_time(i, route.nodes, route.demand, available_nodes, curr_best_time, time_matrix, m.capacity)
                if best_opt is not None:
                    node_added = best_opt[0]
                    curr_best_time = best_opt[1]
                    pos = route.id - 1
                    route_order = i
        if pos < len(routes):  # add the node to an already existing route
            routes[pos].nodes.insert(route_order, node_added)
            routes[pos].time += curr_best_time
            routes[pos].demand += node_added.demand
        else:  # create a new route object containing the [depot, node, depot] route
            routes.append(Route(pos + 1, [depot, node_added, depot], curr_best_time, node_added.demand))
        available_nodes.remove(node_added)
    return routes
