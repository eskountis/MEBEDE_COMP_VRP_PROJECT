import random, math
from matplotlib import pyplot as plt

from Competitional import Analytics
from Competitional.Model import Route, Solution


def worst_fit_with_tsp_nearest(sorted_nodes,m, rcl_size, seed):
    # construct initial solution by using the worst-fit method to create routes and then solving the tsp problem for each route
    routes = apply_worst_fit(sorted_nodes, m.allNodes[0], m.time_matrix, m, rcl_size, seed)
    enhanced_routes = []  # apply tsp nearest algorithm in each route to create a better initial solution
    for r in routes:
        enhanced_routes.append(tsp_nearest(r, m.time_matrix))

    return Solution(0, enhanced_routes)


def route_clustering_with_tsp_nearest(radius_from_depot, m):
    # construct initial solution by assigning the nodes into clusters and then solving the tsp problem for each cluster
    routes = routes_clustering(radius_from_depot, m)
    for r in routes:
        r.nodes.insert(0, m.allNodes[0])
        r.nodes.append(m.allNodes[0])
    enhanced_routes = []  # apply tsp nearest algorithm in each route to create a better initial solution
    for r in routes:
        enhanced_routes.append(tsp_nearest(r, m.time_matrix))
    # routes = create_extra_route(routes, m)
    # enhanced_routes = []  # apply tsp nearest algorithm in each route to create a better initial solution
    # for r in routes:
    #     enhanced_routes.append(tsp_nearest(r, m.time_matrix))

    return Solution(0, enhanced_routes)


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


def create_extra_route(routes, m):
    removed_route = min(routes, key=lambda x: len(x.nodes))
    removed_route = min(routes, key=lambda x: x.time)
    nodes_to_be_transposed = removed_route.nodes[1:-1]
    replacing_route = Route(removed_route.id, [m.allNodes[0]], 0, 0)
    for node in nodes_to_be_transposed:
        if m.dist_matrix[0][node.id] <= 25:  # if the node is close to the center append it to the replacing route
            if replacing_route.demand + node.demand <= m.capacity:
                replacing_route.demand += node.demand
                replacing_route.time += m.time_matrix[replacing_route.nodes[-1].id][node.id]
                nodes_to_be_transposed.remove(node)
                replacing_route.nodes.append(node)

    for route in routes:
        if route.id == removed_route.id:
            continue
        node = route.nodes[1]
        if m.dist_matrix[0][node.id] <= 25:  # if the node is close to the center append it to the replacing route
            if replacing_route.demand + node.demand <= m.capacity:
                replacing_route.demand += node.demand
                replacing_route.time += m.time_matrix[replacing_route.nodes[-1].id][node.id]
                replacing_route.nodes.append(node)
                route.demand -= node.demand
                route.time += m.time_matrix[0][route.nodes[2].id] - m.time_matrix[0][node.id] - m.time_matrix[node.id][route.nodes[2].id]
                route.nodes.remove(node)

    while len(nodes_to_be_transposed) > 0:  # run for the remaining nodes which are not close to the depot
        node = nodes_to_be_transposed.pop(0)
        ids_to_avoid = [removed_route.id]
        for route in routes:
            if route.demand + node.demand > m.capacity:
                ids_to_avoid.append(route.id)
        closest_route = find_closest_route_to_node(node, routes, m, ids_to_avoid)
        closest_route.demand += node.demand
        closest_route.time += m.time_matrix[closest_route.nodes[-2].id][node.id]
        closest_route.nodes.insert(-1, node)

    replacing_route.nodes.append(m.allNodes[0])
    routes.remove(removed_route)
    routes.insert(replacing_route.id, replacing_route)

    return routes


def get_starting_means(means, radius):
    equal_angle_degrees = 360 / means
    angle_cos = math.cos(math.radians(equal_angle_degrees))
    starting_point = (- 1 * radius, 0)
    points = [starting_point]
    # after starting point is set, find its adjacent points by using vector properties
    while len(points) < means:
        if round(starting_point[1], 2) == 0:
            new_point_x = ((radius ** 2) * angle_cos) / starting_point[0]
            new_point_y = math.sqrt((radius ** 2) - new_point_x ** 2)
            new_point_y1 = new_point_y
            new_point_y2 = -1 * new_point_y
            new_point_1 = (new_point_x, new_point_y1)
            new_point_2 = (new_point_x, new_point_y2)
        else:
            a = (starting_point[0]) ** 2 + (starting_point[1]) ** 2
            b = - 2 * (radius ** 2) * angle_cos * starting_point[0]
            c = ((radius ** 2) * angle_cos) ** 2 - (radius ** 2) * (starting_point[1]) ** 2
            D = b ** 2 - 4 * a * c
            if D < 0:
                print("No point found")
                continue
            else:
                new_point_x = (- 1 * b + math.sqrt(D)) / (2 * a)
                new_point_y1 = math.sqrt((radius ** 2) - new_point_x ** 2)
                new_point_y2 = -1 * math.sqrt((radius ** 2) - new_point_x ** 2)
                new_point_1 = (new_point_x, new_point_y1)
                new_point_2 = (new_point_x, new_point_y2)
        insert_pos = int(len(points) / 2) + 1
        if new_point_1 not in points:
            points.insert(insert_pos, new_point_1)
        if new_point_2 not in points:
            points.insert(insert_pos + 1, new_point_2)

        starting_point = new_point_1

    adjusted_points = []
    for i in range(len(points)):
        adjusted_points.append((points[i][0] + 50, points[i][1] + 50))

    return adjusted_points


def get_closest_mean(node, means):
    min_dist = 10 ** 2
    closest_mean = -1
    for i in range(len(means)):
        d_x = (means[i][0] - node.x) ** 2
        d_y = (means[i][1] - node.y) ** 2
        distance = math.sqrt(d_x + d_y)
        if distance < min_dist:
            min_dist = distance
            closest_mean = i
    return closest_mean


def k_means_clustering(m):
    means = m.vehicles
    radius = 20
    means = get_starting_means(means, radius)
    Analytics.visualize_clustering_construction(means, m.customers)
    node_owner = {node.id: -1 for node in m.customers}  # shows the index of the closest mean to that node
    termination_condition = False
    iteration = 0
    while not termination_condition:
        nodes_in_mean_scope = {i: [] for i in range(len(means))}  # key is the mean index and value the nodes that belong to the mean
        change_occurred = False
        for node in m.customers:
            # find the mean closer to the node
            closest_mean = get_closest_mean(node, means)
            nodes_in_mean_scope[closest_mean].append(node)
            # check if a change at the mean to which the node belongs has occurred
            previous_mean = node_owner[node.id]
            if previous_mean != closest_mean:
                change_occurred = True
                node_owner[node.id] = closest_mean

        if not change_occurred:
            termination_condition = True
            continue

        iteration += 1
        Analytics.visualize_clustering_development(iteration, means, nodes_in_mean_scope)

        for i in range(len(means)):
            new_mean_x = sum([n.x for n in nodes_in_mean_scope[i]]) / len(nodes_in_mean_scope[i])
            new_mean_y = sum([n.y for n in nodes_in_mean_scope[i]]) / len(nodes_in_mean_scope[i])
            means[i] = (new_mean_x, new_mean_y)

        iteration += 1
        Analytics.visualize_clustering_development(iteration, means, nodes_in_mean_scope)


def routes_clustering(radius, m):
    means = m.vehicles
    means = get_starting_means(means, radius)
    routes = {i: Route(i, [], 0, 0) for i in range(len(means))}  # key is the mean index and value the nodes that belong to the mean
    node_initial_owner = {node.id: -1 for node in m.customers}  # shows the index of the closest mean to that node
    for node in m.customers:
        # find the mean closer to the node
        closest_mean = get_closest_mean(node, means)
        routes[closest_mean].nodes.append(node)
        routes[closest_mean].demand += node.demand
        node_initial_owner[node.id] = closest_mean
    # Analytics.visualize_k_means_development(1, means, routes)
    untouchable_nodes = []  # the nodes that cause the solution to form infinite loops and shall not be examined again
    iteration = 2
    while True:
        all_routes_valid = True
        for mean, route in routes.items():
            if route.demand <= m.capacity:
                continue
            all_routes_valid = False
            # remove from the overloaded route the most remote node with respect to the rest
            highest_average_dist = 0
            most_remote_node = None
            # find the most alienated node only for the nodes that will not cause infinite loops
            for node in list(filter(lambda x: x.id not in untouchable_nodes, route.nodes)):
                total_dist = sum([m.dist_matrix[node.id][n.id] + m.dist_matrix[n.id][node.id] for n in route.nodes])
                average_dist = total_dist / (2 * (len(route.nodes) - 1))
                if average_dist > highest_average_dist:
                    highest_average_dist = average_dist
                    most_remote_node = node
            # find the next closest mean to insert into
            teased_means = []  # alter the current mean's coordinates so that it won't be selected as the closest one
            for i in routes.keys():
                if i != mean:
                    teased_means.append(means[i])
                else:
                    teased_means.append((means[i][0] * 1000, means[i][1] * 1000))
            closest_mean = get_closest_mean(most_remote_node, teased_means)
            # remove the node from the current route
            route.nodes.remove(most_remote_node)
            route.demand -= most_remote_node.demand
            # insert the node into the new route
            routes[closest_mean].nodes.append(most_remote_node)
            routes[closest_mean].demand += most_remote_node.demand
            # check if the node causes infinite loops
            if node_initial_owner[most_remote_node.id] == closest_mean:
                untouchable_nodes.append(most_remote_node.id)

            # Analytics.visualize_k_means_development(iteration, means, routes)
            # iteration += 1
        if all_routes_valid:
            break

    return list(routes.values())
