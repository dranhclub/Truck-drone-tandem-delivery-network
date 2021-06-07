from scipy.spatial.distance import cdist
from .const import INF
import numpy as np
from math import sqrt


class Task:
    def __init__(self):
        self.best_cost = INF
        self.best_individual = None

    def cost_func(self, individual):
        return 0


class FindingTruckDroneRoute(Task):
    def __init__(self, points, start_point, end_point):
        super().__init__()
        self.points = points
        self.start_point = start_point
        self.end_point = end_point
        self.cached = {}

    def decode(self, genes):
        # TODO:
        return truck_route, drone_route

    def cost_func(self, individual):
        # TODO:

        # Update best cost
        if cost < self.best_cost:
            self.best_cost = cost
            self.best_individual = individual

        return cost
