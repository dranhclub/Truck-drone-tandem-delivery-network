import numpy as np
from .const import INF


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
