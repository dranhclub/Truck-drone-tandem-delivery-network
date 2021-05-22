from point import Point
class Edge:
    def __init__(self, idx, p1: Point, p2: Point):
        self.idx = idx
        self.p1 = p1
        self.p2 = p2
        p1.add_edge(self)
        p2.add_edge(self)

    def __repr__(self):
        return f'({self.idx}:{self.p1.c}-{self.p2.c})'
