import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

INFINITY = 999999999

# P = []  # Set of customer delivery locations
# k_up = len(P)  # Total number of customers or size of P
# dr = 2  # Number of drones per truck

# Generate delivery locations
n = 200
xy_min = [0, 0]
xy_max = [10, 20]
P = np.random.uniform(low=xy_min, high=xy_max, size=(n, 2))


# Show P
# plt.scatter(P[:, 0], P[:, 1])
# plt.show()


def optimize(P):
    pass


def cost(K, P, Ts, Ds, dr):
    pass


def kmeans(k, P):
    # Output:
    C = None  # Set of calculated centroids
    L = np.empty((len(P)), dtype=np.uint64)  # Set of cluster labels
    D = None  # Set of distances

    # Initial cluster assignment
    C = P[np.random.choice(len(P), k, replace=False)]

    while True:
        # Assign customer to closest centroid
        L2 = np.argmin(cdist(P, C), axis=1)

        if np.array_equal(L2, L):
            break
        else:
            L = L2

        # Update centroid of clusters:
        for i in range(k):
            Pi = P[L == i]
            C[i] = np.mean(Pi, axis=0)

    D = np.min(cdist(P, C), axis=1)

    return C, L, D


def genetic(k, C):
    n = k

    # Output
    optimal_route = []
    global_min_dist = None

    # Initialize
    p = 200  # Population size
    POP = np.repeat(np.arange(k).reshape(1,-1), p, axis=0)
    DIST = cdist(C, C)
    global_min_dist = INFINITY
    max_iter = 25 * n ** 9

    def path_dist(path):
        dist = 0
        for i in range(path):
            a = path[i-1]
            b = path[i]
            dist += DIST[a][b]
        return dist

    for iter in range(max_iter):
        # Find min distance route in population
        route_dist = path_dist(POP)
        min_route_idx = np.min(route_dist)
        min_dist = route_dist[min_route_idx]
        if min_dist < global_min_dist:
            # Capture minimum distance and best route
            global_min_dist = min_dist
            optimal_route = POP[min_route_idx]

        # Randomly select 5 routes from population
        POP5 = POP[np.random.choice(p, 5, replace=False)]

        # Find min distance of the 5 routes
        route_dist = path_dist(POP5)
        min_route_idx = np.min(route_dist)
        min_dist = route_dist[min_route_idx]

        # Random shuffle 0:n-1 integers
        r_shuffle = np.random.permutation(n)
        rand1, rand2 = sorted(r_shuffle[1:2])
        rand2, rand3 = sorted(r_shuffle[2:3])
        for k in range(1, 5+1):
            pass


C, L, D = kmeans(5, P)

print("Centroids:", C)
print("Labels:", L)


def plot_kmeans(C, L):
    # import matplotlib.colors as colors
    # colors_list = list(colors._colors_full_map.values())
    colors = ['#f54242', '#0cb000', '#0029b0', '#b0009e', '#b0ad00']

    for i in range(len(C)):
        Pi = P[L == i]
        plt.plot(Pi[:, 0], Pi[:, 1], 'o', color=colors[i], markersize=4, alpha=0.5)

    plt.plot(C[:, 0], C[:, 1], 'ro', markersize=8, alpha=0.8)
    plt.show()


plot_kmeans(C, L)
