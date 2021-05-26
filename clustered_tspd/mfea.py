import re
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata
import copy
import matplotlib.pyplot as plt
from math import sqrt

from clustered_tspd.edge import Edge

INF = 9999999999


class Individual:
    def __init__(self, genes, num_task):
        self.genes = genes
        self.genes_len = len(genes)
        self.has_2_parents = False
        self.parents = []
        self.factorial_cost = [INF] * num_task
        self.factorial_rank = [INF] * num_task
        self.scalar_fitness = INF
        self.skill_factor = None

    def update_skill_factor(self):
        self.skill_factor = np.argmin(self.factorial_rank, axis=0)

    def update_scalar_fitness(self):
        self.scalar_fitness = 1 / np.min(self.factorial_rank)


class Population:
    def __init__(self, genes_length, num_individual):
        self.num_individuals = num_individual
        self.individuals = [None] * num_individual
        self.genes_length = genes_length

    def generate(self, num_task):
        for i in range(self.num_individuals):
            genes = np.random.permutation(self.genes_length)
            individual = Individual(genes, num_task=num_task)
            self.individuals[i] = individual

    def get_best_individual(self, task_index):
        best_idvd = self.individuals[0]
        for idvd in self.individuals:
            if idvd.factorial_cost[task_index] < best_idvd.factorial_cost[task_index]:
                best_idvd = idvd
        return best_idvd

    def __getitem__(self, item):
        return self.individuals[item]

    def __setitem__(self, key, value):
        self.individuals[key] = value

    def __add__(self, other):
        if not isinstance(other, Population):
            raise Exception("Invalid argument")
        new_pop = Population(self.genes_length, self.num_individuals + other.num_individuals)
        new_pop.individuals[:self.num_individuals] = self.individuals
        new_pop.individuals[self.num_individuals:] = other.individuals
        return new_pop


def mutate(p):
    point_1, point_2 = sorted(np.random.choice(len(p), 2, replace=False))
    # Flip
    ret = np.copy(p)
    tmp = p[point_1:point_2 + 1]
    ret[point_1:point_2 + 1] = tmp[::-1]
    return ret


# Partially Mapped Crossover Operator
def _crossover(p1, p2, point_1, point_2):
    def __crossover(p1, p2):
        child = np.empty_like(p1)
        child.fill(-1)
        child[point_1 + 1:point_2 + 1] = p2[point_1 + 1:point_2 + 1]
        for (index, value) in enumerate(child):
            if value != -1:
                continue
            p_value = p1[index]
            while p_value in child:
                p_value = p1[np.where(p2 == p_value)[0][0]]
            child[index] = p_value
        return child

    child_1 = __crossover(p1, p2)
    child_2 = __crossover(p2, p1)
    return child_1, child_2


class MFEA:
    def __init__(self, num_task, genes_length):
        self.num_task = num_task
        self.cost_functions = [None] * num_task
        self.genes_length = genes_length
        self.pop_num = 10
        self.num_loop = 100
        self.rmp = 0.5  # Random mating probability
        self.pop = None

    def set_cost_function(self, task_index, cost_func):
        self.cost_functions[task_index] = cost_func

    def mutate(self, individual):
        child_genes = mutate(individual.genes)
        child = Individual(child_genes, self.num_task)
        child.has_2_parents = False
        child.parents = [individual]
        return child

    def crossover(self, individual_1, individual_2):
        point_1, point_2 = sorted(np.random.choice(len(individual_1.genes) - 1, 2, replace=False))
        genes_1 = individual_1.genes
        genes_2 = individual_2.genes
        child_genes_1, child_genes_2 = _crossover(genes_1, genes_2, point_1, point_2)
        child_1 = Individual(child_genes_1, self.num_task)
        child_1.parents = [individual_1, individual_2]
        child_2 = Individual(child_genes_2, self.num_task)
        child_2.parents = [individual_1, individual_2]
        return child_1, child_2

    def ranking(self, pop):
        num_task = self.num_task
        cost_table = np.zeros((num_task, pop.num_individuals))
        for t in range(num_task):
            for (i, individual) in enumerate(pop.individuals):
                cost_table[t][i] = individual.factorial_cost[t]
        rank_table = np.empty_like(cost_table)
        for t in range(num_task):
            rank_table[t] = rankdata(cost_table[t])
        for i in range(pop.num_individuals):
            pop[i].factorial_rank = rank_table[:, i]

    def run(self):
        self.pop = Population(self.genes_length, self.pop_num)
        pop = self.pop
        pop.generate(self.num_task)
        num_task = self.num_task
        pop_num = self.pop_num
        rmp = self.rmp
        # Evaluate every individual with respect every optimization task
        for t in range(num_task):
            for individual in pop.individuals:
                individual.factorial_cost[t] = self.cost_functions[t](individual)

        # Compute the skill factor of each individual
        self.ranking(pop)
        for individual in pop.individuals:
            individual.update_skill_factor()

        # Main loop
        self.history = []
        for _ in range(self.num_loop):
            # print(f"Sub loop [{_}/{self.num_loop}]")
            # Apply genetic operators on current pop to generate an offspring pop
            offspring_pop = Population(self.genes_length, pop_num)
            count = 0
            while count < pop_num:
                # Assortative mating
                i1, i2 = np.random.choice(pop_num, 2, replace=False)
                idvd1 = pop[i1]
                idvd2 = pop[i2]
                rand = np.random.rand()
                if idvd1.skill_factor == idvd2.skill_factor or rand < rmp:
                    child_1, child_2 = self.crossover(idvd1, idvd2)
                else:
                    child_1 = self.mutate(idvd1)
                    child_2 = self.mutate(idvd2)

                offspring_pop[count] = child_1
                offspring_pop[count + 1] = child_2
                count += 2

            # Evaluate the individuals in offspring-pop for selected optimization tasks only
            for individual in offspring_pop.individuals:
                if individual.has_2_parents:
                    rand = np.random.rand()
                    if rand < 0.5:
                        skill_factor = individual.parents[0].skill_factor
                    else:
                        skill_factor = individual.parents[1].skill_factor
                else:
                    skill_factor = individual.parents[0].skill_factor
                individual.factorial_cost[skill_factor] = self.cost_functions[skill_factor](individual)
                for j in range(num_task):
                    if j != skill_factor:
                        individual.factorial_cost[j] = INF

            # Concatentate offspring-pop and current pop to form and intermediate-pop
            intermediate_pop = pop + offspring_pop

            # Update the scalar fitness and skill factor of every individual in intermediate-pop
            self.ranking(intermediate_pop)
            for individual in intermediate_pop.individuals:
                individual.update_scalar_fitness()
                individual.update_skill_factor()

            # Select the fittest individuals from intermediate-pop to form the next current pop
            fitness_table = np.empty(intermediate_pop.num_individuals)
            for (i, individual) in enumerate(intermediate_pop.individuals):
                fitness_table[i] = individual.scalar_fitness
            selected_indices = np.argsort(fitness_table)[-pop_num:][::-1]
            new_individuals = []
            for i in selected_indices:
                new_individuals.append(intermediate_pop[i])
            pop = Population(self.genes_length, self.pop_num)
            pop.individuals = new_individuals
            self.pop = pop

            # Trace
            bests = [INF] * num_task
            for individual in pop.individuals:
                for i in range(num_task):
                    bests[i] = min(bests[i], individual.factorial_cost[i])
            self.history.append(bests)


def dist(x1, y1, x2, y2):
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


class Task:
    def cost_func(self, individual):
        return 0


class FindingTruckDroneRoute(Task):
    def __init__(self, points, start_point, end_point):
        self.points = points
        self.start_point = start_point
        self.end_point = end_point

    def decode(self, genes):
        points = self.points.copy()
        start_point = self.start_point
        end_point = self.end_point
        points.remove(start_point)
        if start_point != end_point:
            points.remove(end_point)

        # Decode genes
        route = [start_point] + [points[i] for i in genes if i < len(points)] + [end_point]
        return route

    def cost_func(self, individual):
        route = self.decode(individual.genes)
        # TODO: Find optimal route for drone

        # Sum cost
        sum_cost = 0
        for i in range(len(route) - 1):
            a = route[i]
            b = route[i + 1]
            sum_cost += dist(a.x, a.y, b.x, b.y)
        return sum_cost


def display_route(route):
    pass


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
    tspd_mfea.pop_num = 20
    tspd_mfea.num_loop = 20
    tspd_mfea.rmp = 0.5  # Random mating probability

    # Run
    tspd_mfea.run()

    # Add all costs
    sum_cost = 0
    for task_index, task in enumerate(tasks):
        best_individual = tspd_mfea.pop.get_best_individual(task_index)
        sum_cost += best_individual.factorial_cost[task_index]
    for edge in cr:
        sum_cost += dist(edge.p1.x, edge.p1.y, edge.p2.x, edge.p2.y)

    # Show chart
    # plt.plot(tspd_mfea.history)
    # plt.title("MFEA Convergence chart")
    # plt.xlabel("Iteration")
    # plt.ylabel("Best cost")
    # plt.show()

    return sum_cost
