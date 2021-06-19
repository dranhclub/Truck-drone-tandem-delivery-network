import matplotlib.pyplot as plt
from math import sqrt, ceil
from tspd_mfea_pkg.tspd_task import FindingTruckDroneRoute
from mfea_pkg.const import INF
from .tspd_mfea import TSPD_MFEA
import numpy as np
import parameters


def show_convergence_chart(tspd_mfea, tasks):
    num_task = len(tasks)
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


def show_truck_drone_route(tasks):
    num_task = len(tasks)
    plt_size = int(ceil(sqrt(num_task)))
    fig, axs = plt.subplots(plt_size, plt_size)
    fig.suptitle('Routes')
    for task_index, task in enumerate(tasks):
        best_idvd = task.best_individual
        truck_route, drone_route = task.decode(best_idvd.genes)

        row = task_index // plt_size
        col = task_index % plt_size

        # Drone route
        ax = axs[row, col] if plt_size > 1 else axs
        for a, b, c in drone_route:
            ax.plot([a.x, b.x], [a.y, b.y], color='green')
            ax.plot([b.x, c.x], [b.y, c.y], color='green')
            ax.scatter(a.x, a.y, color='blue')
            ax.scatter(b.x, b.y, color='green')
            ax.scatter(c.x, c.y, color='blue')

        # Truck route
        x = [point.x for point in truck_route]
        y = [point.y for point in truck_route]
        ax.plot(x, y, color='blue')
        ax.scatter(x[0], y[0], color='red')
        ax.scatter(x[1:-1], y[1:-1], color='blue')
        ax.scatter(x[-1], y[-1], color='red')
        offset = 0.01
        ax.annotate("s", (x[0] + offset, y[0] + offset))
        ax.annotate("e", (x[-1] + offset, y[-1] + offset))
    plt.show()


def memoize(func):
    cache = dict()

    def memoized_func(clusters, *args):
        if args in cache:
            return cache[args]
        result = func(clusters, *args)
        cache[args] = result
        return result

    return memoized_func


# Minimize cost of truck-drone route when determined cluster route cr
@memoize
def best_cost(clusters, idvd):
    # Find max number of points in the clusters
    max_num_points = 0
    for cluster in clusters:
        max_num_points = max(len(cluster.points), max_num_points)

    # Create solver
    num_task = len(clusters)
    tspd_mfea = TSPD_MFEA(num_task, genes_length=max_num_points + 1)

    # Init start_points and end_points set
    # Note: The following lines of code run correctly if only cluster route is sorted (using dfs before)
    start_points = [None] * num_task
    end_points = [None] * num_task
    # Init last_point
    cr = idvd.genes
    if cr[0].p1.c == 0:
        last_point = cr[0].p1
    elif cr[0].p2.c == 0:
        last_point = cr[0].p2
    else:
        raise Exception("Start of cluster route must be Cluster<0>")
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
    tspd_mfea.pop_num = parameters.MFEA_IDVD_NUM_PER_TASK * num_task
    tspd_mfea.num_loop = parameters.MFEA_NUM_GENERATION
    tspd_mfea.rmp = parameters.MFEA_RMP

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

    # Show result
    show_convergence_chart(tspd_mfea, tasks)
    show_truck_drone_route(tasks)
    # for i in range(num_task):
    #     show_truck_drone_route([tasks[i]])

    return sum_cost
