from typing import List

from clustered_tspd.edge import Edge
from .mfea import MFEA
import matplotlib.pyplot as plt
from math import sqrt, ceil
from .task import FindingTruckDroneRoute
from .const import INF
import numpy as np


def visualize_result(tspd_mfea, tasks):
    num_task = len(tasks)
    # Show convergence chart
    history = np.array(tspd_mfea.history)
    for j in range(num_task):
        mark = False
        for i in range(len(history)):
            if mark:
                history[i][j] = history[i - 1][j]
            else:
                if history[i][j] == INF:
                    mark = True
                    plt.scatter(i, history[i - 1][j])
                    history[i][j] = history[i - 1][j]
    plt.plot(history)
    plt.title("MFEA Convergence chart")
    plt.xlabel("Iteration")
    plt.ylabel("Best cost")
    plt.show()

    # Draw route
    plt_size = int(ceil(sqrt(num_task)))
    fig, axs = plt.subplots(plt_size, plt_size)
    fig.suptitle('Routes')
    for task_index, task in enumerate(tasks):
        best_individual = tspd_mfea.pop.get_best_individual(task_index, tasks[task_index].cost_func)
        route = task.decode(best_individual.genes)
        row = task_index // plt_size
        col = task_index % plt_size
        ax = axs[row, col]
        x = [point.x for point in route]
        y = [point.y for point in route]
        ax.plot(x, y)
        ax.scatter(x[0], y[0], color='gold')
        ax.scatter(x[1:-1], y[1:-1])
        ax.scatter(x[-1], y[-1], color='red')
    plt.show()


# Minimize cost of truck-drone route when determined cluster route cr
def best_cost(clusters, cr: List[Edge]):
    num_task = len(clusters)
    max_num_points = 0
    for cluster in clusters:
        max_num_points = max(len(cluster.points), max_num_points)
    tspd_mfea = MFEA(num_task, genes_length=max_num_points - 1)
    start_points = [None] * num_task
    end_points = [None] * num_task

    if cr[0].p1.c == 0:
        last_point = cr[0].p1
    elif cr[0].p2.c == 0:
        last_point = cr[0].p2
    else:
        raise Exception("Start of cluster route must be Cluster<0>")

    # Note: The following lines of code run correctly if only cluster route is sorted (using dfs before)
    for edge in cr:
        next_point = edge.p1 if last_point.c == edge.p2.c else edge.p2
        end_points[last_point.c] = last_point
        start_points[next_point.c] = next_point
        last_point = next_point

    # Create tasks
    tasks = []
    for i in range(num_task):
        task = FindingTruckDroneRoute(clusters[i].points, start_points[i], end_points[i])
        tasks.append(task)
        tspd_mfea.set_cost_function(i, task.cost_func)

    # Set parameters
    tspd_mfea.pop_num = 3 * num_task
    tspd_mfea.num_loop = 20
    tspd_mfea.rmp = 0.5  # Random mating probability

    # Run
    tspd_mfea.run()

    # Add all costs
    sum_cost = 0
    for task_index, task in enumerate(tasks):
        sum_cost += task.best_cost
    for edge in cr:
        def dist(x1, y1, x2, y2):
            return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        sum_cost += dist(edge.p1.x, edge.p1.y, edge.p2.x, edge.p2.y)

    # visualize_result(tspd_mfea, tasks)
    return sum_cost
