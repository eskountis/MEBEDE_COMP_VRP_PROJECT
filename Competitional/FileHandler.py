import csv


from Competitional.Model import Solution, Route


def store_solution(sol, filename):
    with open(filename, "w") as f:
        f.write(sol.printed_form())
    f.close()


def fetch_solutions(filename, nodes):
    sols = []
    i = 0
    rcl_size = 0
    seed = 0
    with open(filename, "r") as f:
        routes = []
        for line in f:
            elements = line.rstrip().split()
            if len(elements) == 0:
                routes = []
            elif elements[0] == 'Rcl_size:':
                rcl_size, seed = None, None
                if elements[1] != "None":
                    rcl_size = int(elements[1])
                    seed = int(elements[3])
            elif elements[0] == 'obj:':
                sol = Solution(i, routes, rcl_size, seed)
                sols.append(sol)
                i += 1
            else:
                route_id = int(elements[0][:-1])
                route_nodes = [nodes[int(n)] for n in elements[1].split(",")]
                route_time = float(elements[2])
                route_demand = int(elements[3])
                routes.append(Route(route_id, route_nodes, route_time, route_demand))
    f.close()

    return sols


def find_most_popular_args(sols):
    arg_freq = {}
    for s in sols:
        for r in s.routes:
            for i in range(0, len(r.nodes) - 2):
                first_node = r.nodes[i]
                second_node = r.nodes[i + 1]
                arg = (first_node.id, second_node.id)
                if arg not in arg_freq.keys():
                    arg_freq[arg] = 1
                else:
                    arg_freq[arg] += 1
    return arg_freq


def map_by_objective(filename, nodes, upper_obj_threshold):
    sols = fetch_solutions(filename, nodes)
    sol_by_objective = {}

    for s in sols:
        if round(s.obj, 2) not in sol_by_objective.keys():
            sol_by_objective[round(s.obj, 2)] = [s]
        else:
            sol_by_objective[round(s.obj, 2)].append(s)

    qualified = []
    for i in range(380, int(upper_obj_threshold * 100) + 1):
        rounded_obj = i / 100
        if rounded_obj in sol_by_objective.keys():
            for s in sol_by_objective[rounded_obj]:
                qualified.append(s)

    return qualified


def store_in_csv(matrices, filename):
    with open(filename + ".csv", "w", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        for row in range(len(matrices[0])):  # all matrices elements are of the same size
            writer.writerow([matrices[column][row] for column in range(len(matrices))])
    f.close()

