import matplotlib.pyplot as plt
import numpy as np


def generate():
    # Generate clustered points
    def gen_by_cluster(num_cluster, aprxmt_num_point_per_cluster):
        xy_min = [0, 0]
        xy_max = [150, 150]
        centroids = np.random.uniform(low=xy_min, high=xy_max, size=(num_cluster, 2))
        cov = [[7, 0], [0, 7]]
        X = []
        cluster_id = []
        rand_size = np.random.randint(int(0.7 * aprxmt_num_point_per_cluster), int(1.3 * aprxmt_num_point_per_cluster), num_cluster)
        for i in range(num_cluster):
            X.append(np.random.multivariate_normal(centroids[i], cov, rand_size[i]))
            cluster_id += [[i]] * rand_size[i]

        X = np.concatenate(X)
        return np.concatenate((cluster_id, X), axis=1)

    # TODO: Generate edges

    points = gen_by_cluster(10, 7)
    return points, []

def readfile():
    with open("clustered_tspd_data.txt") as f:
        lines = f.readlines()
        num_cluster = 0
        for line in lines:
            if line == '\n':
                num_cluster += 1

        num_points = 0
        l = 0
        cluster = 0
        points = []
        for i in range(num_cluster):
            while lines[l] != '\n':
                num = lines[l].split(" ")
                points.append((cluster, float(num[0]), float(num[1])))
                num_points += 1
                l += 1
            l += 1
            cluster += 1

        cluster_edges = []
        while l < len(lines):
            num = lines[l].split(" ")
            cluster_edges.append((int(num[0]), int(num[1])))
            l += 1

        return np.array(points, dtype='i4, f4, f4'), np.array(cluster_edges)


def show(points, cluster_edges):
    for p1, p2 in cluster_edges:
        p1 = points[p1]
        p2 = points[p2]
        plt.plot([p1[1], p2[1]], [p1[2], p2[2]], color='blue')
    for p in points:
        plt.scatter(p[1], p[2], color='black')
    plt.show()


# show(*readfile())

# show(*generate())
