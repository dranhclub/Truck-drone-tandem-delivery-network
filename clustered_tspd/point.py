from math import sqrt

from cytoolz import memoize


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
        return f'(id:{self.idx}, cluster:{self.c}, x:{self.x}, y:{self.y})'


# def memoize(func):
#     cache = dict()
#
#     def memoized_func(*args):
#         if args in cache:
#             return cache[args]
#         result = func(*args)
#         cache[args] = result
#         return result
#
#     return memoized_func

@memoize
def dist(p1: Point, p2: Point):
    return sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)