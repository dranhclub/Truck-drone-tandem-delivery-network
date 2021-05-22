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
        return f'({self.idx} {self.c} {self.x} {self.y})'
