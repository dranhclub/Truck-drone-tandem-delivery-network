from .const import INF
from abc import ABC, abstractmethod


class Task(ABC):
    def __init__(self):
        self.best_cost = INF
        self.best_individual = None

    @abstractmethod
    def cost_func(self, individual):
        return NotImplemented
