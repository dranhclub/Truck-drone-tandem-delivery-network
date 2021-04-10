import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

INFINITY = 999999999
np.random.seed(0)
# P = []  # Set of customer delivery locations
# k_up = len(P)  # Total number of customers or size of P
# dr = 2  # Number of drones per truck

# Generate delivery locations
n = 250
xy_min = [0, 0]
xy_max = [10, 20]
P = np.random.uniform(low=xy_min, high=xy_max, size=(n, 2))


# Show P
plt.scatter(P[:, 0], P[:, 1])
plt.show()


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


def genetic(C):
    n = len(C)

    # Output
    optimal_route = []
    global_min_dist = None

    # Initialize
    pop_size = 200  # Population size
    POP = np.empty((pop_size, n), dtype=np.uint8)
    for i in range(pop_size):
        POP[i] = np.random.permutation(n)
    DIST = cdist(C, C)
    global_min_dist = INFINITY

    # Loop
    max_iter = 25 * n
    print("max_iter=", max_iter)
    min_dist_by_iter = np.zeros(max_iter)  # save min_dist for plotting
    for iter in range(max_iter):
        if iter % 200 == 0: print("iter=", iter)
        # Calculate total distance of routes
        route_dist = np.zeros(pop_size, dtype=np.float64)
        for i in range(pop_size):
            route_dist[i] = 0
            route = POP[i]
            for j, _ in enumerate(route):
                a = route[j - 1]
                b = route[j]
                route_dist[i] += DIST[a][b]

        # Find best route in population
        min_route_idx = np.argmin(route_dist)
        min_dist = route_dist[min_route_idx]
        min_dist_by_iter[iter] = min_dist  # save min_dist for plotting
        if min_dist < global_min_dist:
            # Capture minimum distance and best route
            global_min_dist = min_dist
            optimal_route = POP[min_route_idx]

        # Genetic algorithm operators
        random_order = np.random.permutation(pop_size)
        new_pop = np.empty((pop_size, n), dtype=np.uint8)
        for p in range(3, pop_size, 4):
            routes = POP[random_order[p - 3:p + 1], :]
            dists = route_dist[random_order[p - 3:p + 1]]
            min_dist_idx = np.argmin(dists)
            min_route = routes[min_dist_idx]
            (a, b) = sorted(np.random.choice(n, 2, replace=False))
            tmp_pop = np.empty((4, n), dtype=np.uint8)
            for k in range(4):
                tmp_pop[k] = min_route
                if k == 1:  # Flip
                    tmp = tmp_pop[k, a:b + 1]
                    tmp_pop[k, a:b + 1] = tmp[::-1]
                elif k == 2:  # Swap
                    tmp_pop[k, [a, b]] = tmp_pop[k, [b, a]]
                elif k == 3:  # Slide
                    ida = [m for m in range(a + 1, b + 1)] + [a]
                    tmp_pop[k, a:b + 1] = tmp_pop[k, ida]
                else:  # Do Nothing
                    pass
            new_pop[p - 3:p + 1] = tmp_pop

        POP = new_pop

    plt.plot(min_dist_by_iter)
    plt.show()
    return optimal_route, global_min_dist


C, L, D = kmeans(50, P)

print("Centroids:", C)
print("Labels:", L)


def plot_kmeans(C, L):
    import matplotlib.colors as colors
    colors = list(colors._colors_full_map.values())
    # colors = ['#f54242', '#0cb000', '#0029b0', '#b0009e', '#b0ad00']

    for i in range(len(C)):
        Pi = P[L == i]
        plt.plot(Pi[:, 0], Pi[:, 1], 'o', color=colors[i], markersize=4, alpha=0.5)

    plt.plot(C[:, 0], C[:, 1], 'ro', markersize=8, alpha=0.8)
    plt.show()


plot_kmeans(C, L)


optimal_route, optimal_dist = genetic(C)
print("Optimal route:", optimal_route)
print("Optimal dist", optimal_dist)
