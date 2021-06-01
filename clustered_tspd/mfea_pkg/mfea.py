import numpy as np
from scipy.stats import rankdata
from .individual import Individual
from .pop import Population

from .const import INF


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
        # Evaluate every individual.py with respect every optimization task
        for t in range(num_task):
            for individual in pop.individuals:
                individual.factorial_cost[t] = self.cost_functions[t](individual)

        # Compute the skill factor of each individual.py
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

            # Update the scalar fitness and skill factor of every individual.py in intermediate-pop
            self.ranking(intermediate_pop)
            for individual in intermediate_pop.individuals:
                individual.update_scalar_fitness()
                individual.update_skill_factor()

            # Select the fittest individuals from intermediate-pop to form the next current pop
            # TODO: Need a smarter selection
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
            # for individual.py in pop.individuals:
            #     for i in range(num_task):
            #         bests[i] = min(bests[i], individual.py.factorial_cost[i])
            # if INF in bests:
            #     print("Best:", bests)
            #     raise Exception("Some task is not evaluated")

            for task_index in range(num_task):
                best_idvd = self.pop.get_best_individual(task_index, self.cost_functions[task_index])
                bests[task_index] = best_idvd.factorial_cost[task_index]
            self.history.append(bests)
