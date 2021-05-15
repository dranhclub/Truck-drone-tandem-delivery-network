import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

np.random.seed(2)


def generate_points_with_min_distance(n, shape, min_dist):
    # compute grid shape based on number of points
    width_ratio = shape[1] / shape[0]
    num_y = np.int32(np.sqrt(n / width_ratio)) + 1
    num_x = np.int32(n / num_y) + 1

    # create regularly spaced neurons
    x = np.linspace(0., shape[1] - 1, num_x, dtype=np.float32)
    y = np.linspace(0., shape[0] - 1, num_y, dtype=np.float32)
    coords = np.stack(np.meshgrid(x, y), -1).reshape(-1, 2)

    # compute spacing
    init_dist = np.min((x[1] - x[0], y[1] - y[0]))

    # perturb points
    max_movement = (init_dist - min_dist) / 2
    noise = np.random.uniform(low=-max_movement,
                              high=max_movement,
                              size=(len(coords), 2))
    coords += noise

    return coords


def generate():
    # np.random.seed(1)
    # Generate clustered points
    def gen_by_cluster(aprxmt_num_cluster, aprxmt_num_point_per_cluster):
        xy_min = [0, 0]
        xy_max = [150, 150]
        # centroids = np.random.uniform(low=xy_min, high=xy_max, size=(num_cluster, 2))
        centroids = generate_points_with_min_distance(n=aprxmt_num_cluster, shape=(150, 150), min_dist=10)
        num_cluster = len(centroids)
        cov = [[7, 0], [0, 7]]
        X = []
        rand_size = np.random.randint(int(0.7 * aprxmt_num_point_per_cluster), int(1.3 * aprxmt_num_point_per_cluster),
                                      num_cluster)
        for i in range(num_cluster):
            for point in np.random.multivariate_normal(centroids[i], cov, rand_size[i]):
                x, y = point
                X.append((i, x, y))

        X = np.array(X, dtype='i4, f4, f4')
        return X, centroids

    # Generate edges
    points, centroids = gen_by_cluster(7, 7)
    edges = []
    for i in range(len(centroids)):
        centr = centroids[i]
        others = [j for j in range(len(centroids)) if i != j]
        dists = []
        for j in others:
            other = centroids[j]
            dist = (centr[0] - other[0]) ** 2 + (centr[1] - other[1]) ** 2
            dists.append(dist)
        argsorted = np.argsort(np.array(dists))
        n = np.random.randint(0, len(others))
        for j in range(n):
            edges.append([i, others[argsorted[j]]])

    edges = np.sort(edges)
    used = []
    cluster_edge = []
    for edge in edges:
        if edge in used:
            continue
        c1 = edge[0]
        c2 = edge[1]
        p_of_c1 = []
        p_of_c2 = []
        id_p_of_c1 = []
        id_p_of_c2 = []
        for i, p in enumerate(points):
            if p[0] == c1:
                p_of_c1.append([p[1], p[2]])
                id_p_of_c1.append(i)
            if p[0] == c2:
                p_of_c2.append([p[1], p[2]])
                id_p_of_c2.append(i)
        p_of_c1 = np.array(p_of_c1)
        p_of_c2 = np.array(p_of_c2)
        r = np.random.rand()
        n = 0 if r < 0.05 else 1 if r < 0.95 else 2
        n = min(n, len(p_of_c1) * len(p_of_c2))  # in case 2 cluster has too few points
        dists = cdist(p_of_c1, p_of_c2)
        indices = np.argsort(dists.ravel())[:n]
        closest = [(ind // dists.shape[1], ind % dists.shape[1]) for ind in indices]
        for i in range(n):
            cluster_edge.append([id_p_of_c1[closest[i][0]], id_p_of_c2[closest[i][1]]])
    return points, cluster_edge


def readfile(filename):
    with open(filename) as f:
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


def save_to_file(points, cluster_edges, filename):
    with open(filename, 'x') as f:
        for i in range(len(points) - 1):
            if points[i][0] == points[i + 1][0]:
                f.write(f'{points[i][1]} {points[i][2]}\n')
            else:
                f.write(f'{points[i][1]} {points[i][2]}\n\n')
        f.write(f'{points[len(points) - 1][1]} {points[len(points) - 1][2]}\n\n')
        for edge in cluster_edges:
            f.write(f'{edge[0]} {edge[1]}\n')


def show(points, cluster_edges):
    for p1, p2 in cluster_edges:
        p1 = points[p1]
        p2 = points[p2]
        plt.plot([p1[1], p2[1]], [p1[2], p2[2]], color='blue')
    for p in points:
        plt.scatter(p[1], p[2], color='black')
    plt.show()

# show(*readfile("mydata.txt"))


# show(*generate())
# save_to_file(*generate(), 'mydata.txt')
