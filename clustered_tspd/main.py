from typing import List

from data import readfile, generate
import numpy as np
import matplotlib.pyplot as plt
from point import Point
from edge import Edge
from cluster import Cluster
from tspd_mfea_pkg.runner import best_cost


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
            cluster_1 = route[i]
            cluster_2 = route[i + 1 if i < len(route) - 1 else 0]

            def match_edge():
                for edge1 in cluster_1.edges:
                    for edge2 in cluster_2.edges:
                        if edge1 == edge2:
                            edge_list.append(edge1)
                            return

            match_edge()
        edge_lists.append([edge for edge in edge_list])
    return edge_lists


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


def crossover_cluster_route(clusters, cr1: List[Edge], cr2: List[Edge]):
    union_edges = list(set(cr1) | set(cr2))
    temp_clusters = [Cluster(i) for i in range(len(clusters))]
    union_points = set()
    for edge in union_edges:
        union_points.add(edge.p1)
        union_points.add(edge.p2)
    union_points = list(union_points)
    for point in union_points:
        if point not in temp_clusters[point.c].points:
            temp_clusters[point.c].points.append(point)
    for edge in union_edges:
        c1 = edge.p1.c
        c2 = edge.p2.c
        temp_clusters[c1].edges.append(edge)
        temp_clusters[c2].edges.append(edge)
        if temp_clusters[c1] not in temp_clusters[c2].neighbors:
            temp_clusters[c2].neighbors.append(temp_clusters[c1])
        if temp_clusters[c2] not in temp_clusters[c1].neighbors:
            temp_clusters[c1].neighbors.append(temp_clusters[c2])
    return generate_cluster_route_using_dfs(temp_clusters, 1)[0], generate_cluster_route_using_dfs(temp_clusters, 1)[0]


class GA_Idvd:
    def __init__(self, genes):
        self.genes = genes
        self.cost = None


if __name__ == '__main__':
    # points, edges, clusters = prepare_data("mydata_9_7.txt")
    points, edges, clusters = prepare_data(None, gen=(6, 15))
    cr_pop_num = 20
    cr_rmp = 0.5  # Random mating probability
    main_loop_num = 20

    # Init cluster route population
    print("Init cluster route population")
    cr_pop = []  # cluster route pop
    for i in range(cr_pop_num):
        route = generate_cluster_route_using_dfs(clusters, 1)[0]
        print(f"Random generate [{i}/{cr_pop_num}] route with length {len(clusters)}: {route}")
        cr_pop.append(GA_Idvd(route))
        # display_route(points, edges, route)

    # Main loop
    history = []
    print("Running main loop")
    for loop_iter in range(main_loop_num):
        print(f"Main loop [{loop_iter}/{main_loop_num}]")
        # Apply GA operator on cr_pop to get cr_offspr_pop
        cr_offspr_pop = []
        count = 0
        while count < cr_pop_num:
            r_idx1, r_idx2 = np.random.choice(cr_pop_num, 2, replace=False)
            cr_parent_1 = cr_pop[r_idx1].genes
            cr_parent_2 = cr_pop[r_idx2].genes
            rand = np.random.rand()
            if rand < cr_rmp:
                child_1, child_2 = crossover_cluster_route(clusters, cr_parent_1, cr_parent_2)
            else:
                child_1 = mutate_cluster_route(edges, cr_parent_1)
                child_2 = mutate_cluster_route(edges, cr_parent_2)
            cr_offspr_pop.append(GA_Idvd(child_1))
            cr_offspr_pop.append(GA_Idvd(child_2))
            count += 2

        # Evaluate every individual in off_spring pop
        for idvd in cr_offspr_pop:
            idvd.cost = best_cost(clusters, idvd.genes)

        # Merge into a intermediate pop
        cr_intermediate_pop = cr_pop + cr_offspr_pop

        # cr_pop = top best cluster route in cr_intermediate_pop
        cr_cost_table = np.empty(len(cr_intermediate_pop))
        for i, idvd in enumerate(cr_intermediate_pop):
            cr_cost_table[i] = idvd.cost
        rank = np.argsort(cr_cost_table)
        new_pop = []
        for i in rank[:cr_pop_num]:
            new_pop.append(cr_intermediate_pop[i])
        cr_pop = new_pop

        history.append(cr_pop[0].cost)

    # Show result
    print("Best cost:", cr_pop[0].cost)
    print("Best cluster route:", cr_pop[0].genes)
    display_route(points, edges, cr_pop[0].genes)
    plt.plot(history)
    plt.title("Convergence chart")
    plt.xlabel("Iteration")
    plt.ylabel("Best cost")
    plt.show()
