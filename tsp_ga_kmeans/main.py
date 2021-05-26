import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from math import ceil

INFINITY = 999999999
np.random.seed(0)

# Generate delivery locations
n = 50
xy_min = [0, 0]
xy_max = [30, 30]


# P = np.random.uniform(low=xy_min, high=xy_max, size=(n, 2))


# Generate delivery locations by cluster
def gen_by_cluster(num_cluster):
    centroids = np.random.uniform(low=xy_min, high=xy_max, size=(num_cluster, 2))
    cov = [[1, 0], [0, 1]]
    X = []
    n2 = n
    k = n // num_cluster
    for centroid in centroids:
        if (n2 > k):
            X.append(np.random.multivariate_normal(centroid, cov, k))
            n2 -= k
        else:
            X.append(np.random.multivariate_normal(centroid, cov, n2))
    return np.concatenate(X)


def optimize(P):
    k_up = len(P)  # Total number of customers or size of P
    dr = 4  # Number of drones per truck

    # Initialize
    max_iter = 100
    best_time = INFINITY
    optimal_k = k_up
    Ts = 35
    Ds = Ts * 1.5
    k_low = 2
    K = np.empty(max_iter, dtype=np.int8)
    K[0] = int(ceil(k_up / 5))
    J = np.zeros(max_iter, dtype=np.float64)
    J[0] = cost(K[0], P, Ts, Ds, dr)
    alpha = 10
    g = 1

    # Loop
    for i in range(max_iter):
        K[i + 1] = K[i] - alpha * g
        J[i + 1] = cost(K[i + 1], P, Ts, Ds, dr)
        # print("K[%d]=%d" % (i+1, K[i+1]))
        # print("J[%d]=%d" % (i+1, J[i+1]))
        dJ = J[i + 1] - J[i]
        dK = K[i + 1] - K[i]
        if dK == 0:
            break
        g = dJ / dK

        # Capture best time and optimal k
        if J[i + 1] < best_time:
            best_time = J[i + 1]
            optimal_k = K[i + 1]

    print("J=", J)
    print("J shape=", J.shape)
    plt.plot(J)
    plt.show()
    return best_time, optimal_k


def try_k(P):
    dr = 4  # Number of drones per truck

    # Initialize
    max_iter = n // 2
    Ts = 35
    Ds = Ts * 1.5
    K = np.empty(max_iter, dtype=np.int8)
    J = np.zeros(max_iter, dtype=np.float64)
    for k in range(2, max_iter):
        J[k] = cost(k, P, Ts, Ds, dr)

    print("J=", J)
    print("J shape=", J.shape)
    plt.xlabel("K")
    plt.ylabel("cost")
    plt.plot(list(range(2,max_iter)), J[2:max_iter])
    plt.show()


def cost(K, P, Ts, Ds, dr):
    max_iter = 1
    sum_total_time = 0
    for i in range(max_iter):
        C, L, D = kmeans(K, P)
        sum_D = 2 * np.sum(D)
        route, route_dist = genetic(C)
        sum_total_time += route_dist / Ts + (1 / dr) * sum_D / Ds
    total_time_avg = sum_total_time / max_iter
    return total_time_avg


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

    # plt.plot(min_dist_by_iter)
    # plt.title("Num cluster k=" + str(n))
    # plt.xlabel("Iteration")
    # plt.ylabel("Min distance")
    # plt.show()
    return optimal_route, global_min_dist


def plot_kmeans(C, L):
    import matplotlib.colors as colors
    colors = list(colors._colors_full_map.values())
    # colors = ['#f54242', '#0cb000', '#0029b0', '#b0009e', '#b0ad00']

    for i in range(len(C)):
        Pi = P[L == i]
        plt.plot(Pi[:, 0], Pi[:, 1], 'o', color=colors[i], markersize=4, alpha=0.5)

    plt.plot(C[:, 0], C[:, 1], 'ro', markersize=8, alpha=0.8)
    plt.title("Kmeans clustering")
    plt.show()


def test_kmeans(P):
    C, L, D = kmeans(20, P)
    plot_kmeans(C, L)


def test_genetic(P):
    C, L, D = kmeans(20, P)
    plot_kmeans(C, L)
    optimal_route, optimal_dist = genetic(C)
    print("Optimal route:", optimal_route)
    print("Optimal dist", optimal_dist)


P = gen_by_cluster(3)

# Show P
plt.scatter(P[:, 0], P[:, 1])
plt.show()

# best_time, optimal_k = optimize(P)
# print("Best time:", best_time)
# print("Optimal k:", optimal_k)

# test_genetic(P)

try_k(P)

