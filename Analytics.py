import math
from matplotlib import pyplot as plt
import numpy as np

from Competitional import FileHandler
from Competitional.Model import Model


def bar_chart(sol, time_matrix):
    width = 0.35
    stacks = []
    route_with_most_nodes = max(sol.routes, key=lambda x: len(x.nodes))
    for i in range(len(route_with_most_nodes.nodes) - 1):
        stack = []
        for r in sol.routes:
            if i < len(r.nodes) - 1:
                stack.append(time_matrix[r.nodes[i].id][r.nodes[i+1].id])
            else:
                stack.append(0)
        stacks.append(stack)

    # for i in range(len(stacks)):
    #     if i == 0:
    #         plt.bar(np.arange(len(sol.routes)), stacks[i], width)
    #     else:
    #         plt.bar(np.arange(len(sol.routes)), stacks[i], width, bottom=stacks[i-1])

    sorted_routes = sorted(sol.routes, key=lambda x: x.time)
    plt.bar(np.arange(len(sol.routes)), [r.time for r in sorted_routes], width)

    plt.ylabel('Cost')
    plt.xticks(np.arange(len(sol.routes)), np.arange(len(sol.routes)))
    plt.yticks(np.arange(0, 6, 0.2))

    plt.show()


def find_outliers(sol):
    sol.update_median_and_st_dev()
    normal_values = []
    outliers = []
    for r in sol.routes:
        if sol.median - 1 * sol.st_dev < r.time < sol.median + 1 * sol.st_dev:
            normal_values.append(r)
        else:
            outliers.append(r)
    sol.print()
    print(sol.median, sol.st_dev)
    for r in outliers:
        print([n.id for n in r.nodes], r.time)
    # bar_chart(sol, time_matrix)


def visualize_sol_evolvement(objectives):
    plt.plot(np.arange(len(objectives)), objectives, '.-')

    plt.show()
    plt.close()
