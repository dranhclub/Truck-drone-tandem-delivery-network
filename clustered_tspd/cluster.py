class Cluster:
    def __init__(self, idx):
        self.idx = idx
        self.points = []
        self.edges = []
        self.neighbors = []

    def __repr__(self):
        return f'Cluster<{self.idx}>'
