from typing import List

from data import readfile, generate
import numpy as np
import matplotlib.pyplot as plt
import copy
from math import sqrt
import random
from point import Point
from edge import Edge
from cluster import Cluster


def prepare_data(filename, gen=None):
    if gen:
        (aprxmt_num_cluster, aprxmt_num_point_per_cluster) = gen
        r_points, r_edges = generate(aprxmt_num_cluster, aprxmt_num_point_per_cluster)
    else:
        r_points, r_edges = readfile(filename)
    points = []
    edges = []
    num_cluster = r_points[-1][0] + 1
    clusters = [Cluster(i) for i in range(num_cluster)]
    for i, (c, x, y) in enumerate(r_points):
        point = Point(i, c, x, y)
        points.append(point)
        clusters[c].points.append(point)
    for i, (idp1, idp2) in enumerate(r_edges):
        edge = Edge(i, points[idp1], points[idp2])
        edges.append(edge)
        c1 = edge.p1.c
        c2 = edge.p2.c
        clusters[c1].edges.append(edge)
        clusters[c2].edges.append(edge)
        if clusters[c2] not in clusters[c1].neighbors:
            clusters[c1].neighbors.append(clusters[c2])
        if clusters[c1] not in clusters[c2].neighbors:
            clusters[c2].neighbors.append(clusters[c1])
    display_route(points, edges, [])
    return points, edges, clusters


def generate_cluster_route_using_dfs(clusters, prefer_num_route):
    routes = []

    def dfs(route):
        nonlocal routes
        last_cluster = route[len(route) - 1]
        if len(route) == len(clusters):
            if clusters[0] in last_cluster.neighbors:
                routes.append([cluster for cluster in route])
            return
        ran_idx = np.random.permutation(len(last_cluster.neighbors))
        for i in ran_idx:
            cluster = last_cluster.neighbors[i]
            if cluster not in route:
                route.append(cluster)
                dfs(route)
                route.pop()
                if len(routes) == prefer_num_route:
                    return

    route = [clusters[0]]
    dfs(route)
    edge_lists = []
    for route in routes:
        edge_list = []
        for i in range(len(route)):
            c1 = route[i]
            c2 = route[i+1 if i < len(route) - 1 else 0]

            def match_edge():
                for edge1 in c1.edges:
                    for edge2 in c2.edges:
                        if edge1 == edge2:
                            edge_list.append(edge1)
                            return

            match_edge()
        edge_lists.append([edge for edge in edge_list])
    return edge_lists


def generate_cluster_route(clusters, edges):
    num_cluster = len(clusters)
    tour = np.random.permutation(num_cluster - 1)
    tour = np.concatenate((tour, [num_cluster - 1]))

    def is_connected(cluster1, cluster2):
        cluster1 = clusters[cluster1]
        cluster2 = clusters[cluster2]
        for edge in cluster1.edges:
            if edge.p1.c == cluster2.idx or edge.p2.c == cluster2.idx:
                return True

    def is_valid(tour):
        for i in range(len(tour) - 1):
            if not is_connected(tour[i], tour[i + 1]):
                return False
        if not is_connected(tour[len(tour) - 1], tour[0]):
            return False
        return True

    count = 0
    while not is_valid(tour):
        tour = np.random.permutation(num_cluster - 1)
        tour = np.concatenate((tour, [num_cluster - 1]))
        count += 1
        if count > 100000:
            raise Exception("Cannot generate valid tour")
    route = []
    for i in range(len(tour) - 1):
        cluster1 = clusters[tour[i]]
        cluster2 = clusters[tour[i + 1]]
        for edge in random.sample(cluster1.edges, len(cluster1.edges)):
            if edge.p1.c == cluster2.idx or edge.p2.c == cluster2.idx:
                route.append(edge)
                break
        else:
            raise Exception("Error")
    cluster1 = clusters[tour[len(tour) - 1]]
    cluster2 = clusters[tour[0]]
    for edge in random.sample(cluster1.edges, len(cluster1.edges)):
        if edge.p1.c == cluster2.idx or edge.p2.c == cluster2.idx:
            route.append(edge)
            break
    else:
        raise Exception("Error")

    return route


def display_route(points, edges, route: List[Edge]):
    for p in points:
        plt.scatter(p.x, p.y, color="black")
    for edge in edges:
        plt.plot([edge.p1.x, edge.p2.x], [edge.p1.y, edge.p2.y], color='blue')
    for edge in route:
        plt.plot([edge.p1.x, edge.p2.x], [edge.p1.y, edge.p2.y], color='red')
    plt.show()


def mutate_cluster_route(edges, cr):
    cr_perm = np.random.permutation(cr)
    for edge in cr_perm:
        for edge2 in edges:
            if edge != edge2 and (edge.p1.c == edge2.p1.c and edge.p2.c == edge2.p2.c) or (
                    edge.p1.c == edge2.p2.c and edge.p2.c == edge2.p1.c):
                mutated_cr = []
                for e in cr:
                    if e != edge:
                        mutated_cr.append(e)
                    else:
                        mutated_cr.append(edge2)
                return mutated_cr
    return cr


def crossover_cluster_route(clusters, cr1, cr2):
    union = list(set(cr1) | set(cr2))
    return generate_cluster_route(clusters, union), generate_cluster_route(clusters, union)


def dist(x1, y1, x2, y2):
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def best_cost(cr):
    # TODO: use MFEA to solve
    sum_cost = 0
    for edge in cr:
        sum_cost += dist(edge.p1.x, edge.p1.y, edge.p2.x, edge.p2.y)
    return sum_cost


if __name__ == '__main__':
    # points, edges, clusters = prepare_data("mydata_9_7.txt")
    points, edges, clusters = prepare_data(None, gen=(9, 7))
    cr_pop_num = 20
    cr_rmp = 0.0  # Random mating probability
    main_loop_num = 20
    sub_loop_num = 20

    # Init cluster route population
    print("Init cluster route population")
    cr_pop = []  # cluster route pop
    for i in range(cr_pop_num):
        route = generate_cluster_route_using_dfs(clusters, 1)[0]
        print(route)
        cr_pop.append(route)
        # display_route(points, edges, route)

    # Main loop
    history = []
    for _ in range(main_loop_num):
        # Apply GA operator on cr_pop to get cr_offspr_pop
        cr_offspr_pop = []
        count = 0
        while count < cr_pop_num:
            r_idx1, r_idx2 = np.random.choice(cr_pop_num, 2, replace=False)
            cr_parent_1 = cr_pop[r_idx1]
            cr_parent_2 = cr_pop[r_idx2]
            rand = np.random.rand()
            if rand < cr_rmp:
                child_1, child_2 = crossover_cluster_route(clusters, cr_parent_1, cr_parent_2)
            else:
                child_1 = mutate_cluster_route(edges, cr_parent_1)
                child_2 = mutate_cluster_route(edges, cr_parent_2)
            cr_offspr_pop.append(child_1)
            cr_offspr_pop.append(child_2)
            count += 2

        # Merge into a intermediate pop
        cr_intermediate_pop = cr_pop + cr_offspr_pop

        # Evaluate every cluster route in cr_intermediate_pop (1)
        cr_cost_table = np.empty(len(cr_intermediate_pop))
        for i, cr in enumerate(cr_intermediate_pop):
            cr_cost_table[i] = best_cost(cr)

        # cr_pop = top best cluster route in cr_intermediate_pop
        rank = np.argsort(cr_cost_table)
        new_pop = []
        for i in rank[:cr_pop_num]:
            new_pop.append(cr_intermediate_pop[i])
        cr_pop = new_pop

        history.append(best_cost(cr_pop[0]))
    # Show result
    for cr in cr_pop:
        print(cr)
        print(best_cost(cr))
    display_route(points, edges, cr_pop[0])
    plt.plot(history)
    plt.show()
    print(cr_pop[0])
    print(mutate_cluster_route(edges, cr_pop[0]))

'''
Main thread:
    init cr_pop (cluster route pop)
    while stop condition is not satisfied:
        apply GA operator on cr_pop to get cr_offspr_pop
        cr_intermediate_pop = cr_pop + cr_offpsr_pop
        evaluate every cluster route in cr_intermediate_pop (1)
        cr_pop = top best cluster route in cr_intermediate_pop

(1) Evaluate cluster route:
    Each cluster => one task of TSP-D
    Merge into one big task and solve it by using MFEA, then get a optimal route
    Sum all cost to evaluate
'''
