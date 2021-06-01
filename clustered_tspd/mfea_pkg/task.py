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
        points = self.points.copy()
        start_point = self.start_point
        end_point = self.end_point
        points.remove(start_point)
        if start_point != end_point:
            points.remove(end_point)

        route = [start_point] + [points[i] for i in genes if i < len(points)] + [end_point]
        return route

    # Split route into truck + drone route
    def split_algorithm(self, route):
        truck_speed = drone_speed = 1
        X = np.array([[p.x, p.y] for p in route])
        truck_cost = cdist(X, X) / truck_speed
        drone_cost = truck_cost / drone_speed

        M = np.empty(len(route), dtype=np.float32)
        M.fill(-1)
        prev = np.empty(len(route), dtype=np.int8)
        prev.fill(-1)

        M[0] = 0
        M[1] = truck_cost[0][1]
        prev[0] = -1
        prev[1] = 0

        def min_cost(i):
            if i < 2:
                return M[i]
            if M[i] == -1:
                c1 = min_cost(i - 1) + truck_cost[i - 1][i]
                c2 = min_cost(i - 2) + max(truck_cost[i - 2][i], drone_cost[i - 2][i - 1] + drone_cost[i - 1][i])
                if c1 <= c2:
                    M[i] = c1
                    prev[i] = i - 1
                else:
                    M[i] = c2
                    prev[i] = i - 2
            return M[i]

        bestcost = min_cost(len(route) - 1)

        truck_route = []
        i = len(route) - 1
        while i >= 0:
            truck_route.insert(0, route[i])
            i = prev[i]

        drone_route = []
        for i in range(1, len(route) - 1):
            if route[i] not in truck_route:
                drone_route.append([route[i - 1], route[i], route[i + 1]])

        return bestcost, truck_route, drone_route

    def cost_func(self, individual):
        route = self.decode(individual.genes)
        # use drone
        # cost, truck_route, drone_route = self.split_algorithm(route)
        # return cost

        # not use drone
        cost = 0
        for i in range(len(route) - 1):
            a = route[i]
            b = route[i + 1]
            cost += sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

        # Update best cost
        if cost < self.best_cost:
            self.best_cost = cost
            self.best_individual = individual

        return cost
