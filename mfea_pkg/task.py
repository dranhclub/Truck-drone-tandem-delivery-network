from scipy.spatial.distance import cdist
from .const import INF
import numpy as np
from math import sqrt
from tspd_mfea_pkg.tspd_mfea import fix_color
from clustered_tspd.point import dist


class Task:
    def __init__(self):
        self.best_cost = INF
        self.best_individual = None

    def cost_func(self, individual):
        return 0


class FindingTruckDroneRoute(Task):
    def __init__(self, points, start_point, end_point):
        super().__init__()
        points = points.copy()
        points.remove(start_point)
        if start_point != end_point:
            points.remove(end_point)
        self.points_in_order = [start_point] + points + [end_point]

    def decode(self, genes):
        gen_order, gen_color = genes
        N = len(self.points_in_order)
        order = [0] + [i for i in gen_order if 0 < i < N - 1] + [N - 1]
        color = fix_color([gen_color[i] for i in range(N)])

        # Get truck route
        truck_route = []
        for i in range(N):
            if color[i] != "G":
                truck_route.append(self.points_in_order[order[i]])

        # Get drone route
        drone_route = []

        last_color = None
        launch_point = -1
        delivery_point = -1

        for i in range(N):
            if color[i] == "R":
                if last_color == "G":
                    a = self.points_in_order[order[launch_point]]
                    b = self.points_in_order[order[delivery_point]]
                    c = self.points_in_order[order[i]]
                    drone_route.append([a, b, c])
                launch_point = i
            elif color[i] == "G":
                delivery_point = i
            last_color = color[i]
        return truck_route, drone_route

    def cost_func(self, individual):
        truck_route, drone_route = self.decode(individual.genes)
        drone_speed = 1
        truck_speed = 1

        segment_cost = []
        for a, b, c in drone_route:
            drone_cost = (dist(a, b) + dist(b, c)) / drone_speed
            id_a = truck_route.index(a)
            id_c = truck_route.index(c)
            truck_cost = 0
            for i in range(id_a, id_c):
                truck_cost += dist(truck_route[i], truck_route[i + 1]) / truck_speed
            segment_cost.append(max(drone_cost, truck_cost))

        cost = 0
        j = 0
        i = 0
        while i < len(truck_route) - 1:
            if j < len(drone_route) and drone_route[j][0] == truck_route[i]:
                cost += segment_cost[j]
                i = truck_route.index(drone_route[j][2])
                j += 1
            else:
                cost += dist(truck_route[i], truck_route[i + 1]) / truck_speed
                i += 1

        # Update best cost
        if cost < self.best_cost:
            self.best_cost = cost
            self.best_individual = individual

        return cost
