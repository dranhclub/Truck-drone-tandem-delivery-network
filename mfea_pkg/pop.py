# import numpy as np
# from .individual import Individual
#
# class Population:
#     def __init__(self, genes_length, num_individual):
#         self.num_individuals = num_individual
#         self.individuals = [None] * num_individual
#         self.genes_length = genes_length
#
#     def generate(self, num_task):
#         for i in range(self.num_individuals):
#             genes = np.random.permutation(self.genes_length)
#             individual = Individual(genes, num_task=num_task)
#             self.individuals[i] = individual
#
#     def get_best_individual(self, task_index, cost_func):
#         best_idvd = self.individuals[0]
#         for idvd in self.individuals:
#             if idvd.factorial_cost[task_index] < best_idvd.factorial_cost[task_index]:
#                 best_idvd = idvd
#         return best_idvd
#
#     def __getitem__(self, item):
#         return self.individuals[item]
#
#     def __setitem__(self, key, value):
#         self.individuals[key] = value
#
#     def __add__(self, other):
#         if not isinstance(other, Population):
#             raise Exception("Invalid argument")
#         new_pop = Population(self.genes_length, self.num_individuals + other.num_individuals)
#         new_pop.individuals[:self.num_individuals] = self.individuals
#         new_pop.individuals[self.num_individuals:] = other.individuals
#         return new_pop