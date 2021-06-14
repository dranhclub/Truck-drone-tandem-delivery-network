from typing import Callable

import numpy as np
from scipy.stats import rankdata
from .individual import Individual
from abc import ABC, abstractmethod
from .const import INF


class MFEA(ABC):
    def __init__(self, num_task, genes_length):
        self.num_task = num_task
        self.cost_functions = [Callable] * num_task
        self.genes_length = genes_length
        self.pop_num = 10
        self.num_loop = 100
        self.rmp = 0.5  # Random mating probability
        self.pop = []

    def __mutate(self, individual):
        child = self.mutate(individual)
        child.has_2_parents = False
        child.parents = [individual]
        return child

    def __crossover(self, individual_1, individual_2):
        child_1, child_2 = self.crossover(individual_1, individual_2)
        child_1.has_2_parents = True
        child_2.has_2_parents = True
        child_1.parents = [individual_1, individual_2]
        child_2.parents = [individual_1, individual_2]
        return child_1, child_2

    def set_cost_function(self, task_index, cost_func):
        self.cost_functions[task_index] = cost_func

    @abstractmethod
    def mutate(self, individual) -> Individual:
        return NotImplemented

    @abstractmethod
    def crossover(self, individual_1, individual_2) -> (Individual, Individual):
        return NotImplemented

    def ranking(self, pop):
        num_task = self.num_task
        cost_table = np.zeros((num_task, len(pop)))
        for t in range(num_task):
            for (i, individual) in enumerate(pop):
                cost_table[t][i] = individual.factorial_cost[t]
        rank_table = np.empty_like(cost_table)
        for t in range(num_task):
            rank_table[t] = rankdata(cost_table[t])
        for i in range(len(pop)):
            pop[i].factorial_rank = rank_table[:, i]

    @abstractmethod
    def generate_pop(self):
        pass

    def get_best_individual(self, task_index):
        best_idvd = None
        for idvd in self.pop:
            if best_idvd == None or idvd.factorial_cost[task_index] < best_idvd.factorial_cost[task_index]:
                best_idvd = idvd
        return best_idvd

    def run(self):
        self.pop = self.generate_pop()
        pop = self.pop
        num_task = self.num_task
        pop_num = self.pop_num

        # Evaluate every individual with respect every optimization task
        for t in range(num_task):
            for individual in pop:
                individual.factorial_cost[t] = self.cost_functions[t](individual)

        # Compute the skill factor of each individual
        self.ranking(pop)
        for individual in pop:
            individual.update_skill_factor()

        # Main loop
        self.history = []
        for _ in range(self.num_loop):
            # print(f"Sub loop [{_}/{self.num_loop}]")
            # Apply genetic operators on current pop to generate an offspring pop
            offspring_pop = []
            count = 0
            while count < pop_num:
                # Assortative mating
                i1, i2 = np.random.choice(pop_num, 2, replace=False)
                idvd1 = pop[i1]
                idvd2 = pop[i2]
                rand = np.random.rand()
                if idvd1.skill_factor == idvd2.skill_factor or rand < self.rmp:
                    child_1, child_2 = self.__crossover(idvd1, idvd2)
                else:
                    child_1 = self.__mutate(idvd1)
                    child_2 = self.__mutate(idvd2)

                offspring_pop += [child_1, child_2]
                count += 2

            # Evaluate the individuals in offspring-pop for selected optimization tasks only
            for individual in offspring_pop:
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

            # Update the scalar fitness and skill factor of every individual.py in intermediate-pop
            self.ranking(intermediate_pop)
            for individual in intermediate_pop:
                individual.update_scalar_fitness()
                individual.update_skill_factor()

            # Select the fittest individuals from intermediate-pop to form the next current pop
            # TODO: Need a smarter selection
            fitness_table = np.empty(len(intermediate_pop))
            for (i, individual) in enumerate(intermediate_pop):
                fitness_table[i] = individual.scalar_fitness
            selected_indices = np.argsort(fitness_table)[-pop_num:][::-1]
            pop = []
            for i in selected_indices:
                pop.append(intermediate_pop[i])
            self.pop = pop

            # Trace
            bests = [INF] * num_task
            for task_index in range(num_task):
                best_idvd = self.get_best_individual(task_index)
                bests[task_index] = best_idvd.factorial_cost[task_index]
            self.history.append(bests)
