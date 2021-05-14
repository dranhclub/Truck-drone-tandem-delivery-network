from typing import List

from data import readfile
import numpy as np
import matplotlib.pyplot as plt
import copy
from math import sqrt


class Point:
    def __init__(self, idx, c, x, y):
        self.idx = idx
        self.c = c
        self.x = x
        self.y = y
        self.edges = []

    def add_edge(self, edge):
        self.edges.append(edge)

    def __repr__(self):
        return f'({self.idx} {self.c} {self.x} {self.y})'


class Edge:
    def __init__(self, idx, p1: Point, p2: Point):
        self.idx = idx
        self.p1 = p1
        self.p2 = p2
        p1.add_edge(self)
        p2.add_edge(self)

    def __repr__(self):
        return f'({self.idx}:{self.p1.idx}-{self.p2.idx})'


def prepare_data():
    r_points, r_edges = readfile()
    points = []
    edges = []
    for i, (c, x, y) in enumerate(r_points):
        points.append(Point(i, c, x, y))
    for i, (idp1, idp2) in enumerate(r_edges):
        edges.append(Edge(i, points[idp1], points[idp2]))
    return points, edges


def generate_cluster_route(num_cluster, edges):
    rand_idx = np.random.permutation(len(edges))
    route = []
    conn_level = [0] * num_cluster
    for i in rand_idx:
        edge = edges[i]
        if conn_level[edge.p1.c] == 2 or conn_level[edge.p2.c] == 2:
            continue

        for e in route:
            if e.p1.c == edge.p1.c and e.p2.c == edge.p2.c:
                break
            if e.p2.c == edge.p1.c and e.p1.c == edge.p2.c:
                break
        else:
            route.append(edge)
            conn_level[edge.p1.c] += 1
            conn_level[edge.p2.c] += 1
    return route


def display_route(points, edges, route: List[Edge]):
    for p in points:
        plt.scatter(p.x, p.y, color="blue")
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


def crossover_cluster_route(num_cluster, cr1, cr2):
    union = list(set(cr1) | set(cr2))
    return generate_cluster_route(num_cluster, union), generate_cluster_route(num_cluster, union)


def dist(x1, y1, x2, y2):
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def best_cost(cr):
    # TODO: use MFEA to solve
    sum_cost = 0
    for edge in cr:
        sum_cost += dist(edge.p1.x, edge.p1.y, edge.p2.x, edge.p2.y)
    return sum_cost


if __name__ == '__main__':
    points, edges = prepare_data()
    num_cluster = points[-1].c + 1
    cr_pop_num = 10
    cr_rmp = 0.5  # Random mating probability
    main_loop_num = 20
    sub_loop_num = 20

    # Init cluster route population
    cr_pop = []  # cluster route pop
    for i in range(cr_pop_num):
        route = generate_cluster_route(num_cluster, edges)
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
                child_1, child_2 = crossover_cluster_route(num_cluster, cr_parent_1, cr_parent_2)
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
    print(history)
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
