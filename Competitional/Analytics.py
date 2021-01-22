import math
from matplotlib import pyplot as plt
import numpy as np

from Competitional import FileHandler
from Competitional.Model import Model


def visualize_clustering_construction(points, all_nodes):
    x_points = [n.x for n in all_nodes]
    y_points = [n.y for n in all_nodes]
    plt.scatter(x_points, y_points)
    mean_x_points = [p[0] for p in points] + [points[-1][0], points[0][0]]
    mean_y_points = [p[1] for p in points] + [points[-1][1], points[0][1]]
    plt.scatter(mean_x_points, mean_y_points, color="black")
    # plt.plot(mean_x_points, mean_y_points, color="black")
    # for p in points:
    #     plt.plot([p[0], 50], [p[1], 50], color="black")
    plt.savefig("0.png")
    plt.close()


def visualize_clustering_development(iteration, means, groups):
    for i in range(len(means)):
        x_points = [n.x for n in groups[i].nodes]
        y_points = [n.y for n in groups[i].nodes]
        plt.plot(x_points, y_points, ".-")
    mean_x_points = [p[0] for p in means]
    mean_y_points = [p[1] for p in means]
    plt.scatter(mean_x_points, mean_y_points, color="black")
    plt.savefig(str(iteration) + ".png")
    plt.close()




def scatter_nodes(all_nodes):
    x_points = [n.x for n in all_nodes]
    y_points = [n.y for n in all_nodes]
    plt.scatter(x_points, y_points)
    plt.show()
    plt.close()


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


def visualize_sol_evolvement(x_axis, objectives):
    plt.plot(x_axis, objectives, '.-')
    plt.show()
    plt.close()
