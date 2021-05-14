import re
import numpy as np
import matplotlib.pyplot as plt


N = 0
mat = []

filename = "p01_d.txt"
with open(filename) as f:
    lines = f.readlines()
    N = len(lines)
    for line in lines:
        mat.append(re.split("\s+", line)[1:N + 1])

    for i in range(N):
        for j in range(N):
            mat[i][j] = int(mat[i][j])

# Genetic algorithm parameters
n_pop = 200
pop = []
numloop = 400
rate_of_mutate = 0.5

# Initialize population
pop = np.empty((n_pop, N), dtype=np.int32)

for i in range(n_pop):
    pop[i] = np.random.permutation(N)
print("Init pop", pop)


def mutate(p):
    point_1, point_2 = sorted(np.random.choice(N, 2, replace=False))
    # Flip
    ret = np.copy(p)
    tmp = p[point_1:point_2 + 1]
    ret[point_1:point_2 + 1] = tmp[::-1]
    return ret


# Partially Mapped Crossover Operator
def _crossover(p1, p2, point_1, point_2):
    def __crossover(p1, p2):
        child = np.empty_like(p1)
        child.fill(-1)
        child[point_1 + 1:point_2 + 1] = p2[point_1 + 1:point_2 + 1]
        for (index, value) in enumerate(child):
            if value != -1:
                continue
            p_value = p1[index]
            while p_value in child:
                p_value = p1[np.where(p2 == p_value)[0][0]]
            child[index] = p_value
        return child

    child_1 = __crossover(p1, p2)
    child_2 = __crossover(p2, p1)
    return child_1, child_2


def crossover(p1, p2):
    point_1, point_2 = sorted(np.random.choice(N - 1, 2, replace=False))
    return _crossover(p1, p2, point_1, point_2)


def cost(p):
    sum = 0
    for i in range(N - 1):
        a = p[i]
        b = p[i + 1]
        sum += mat[a][b]
    sum += mat[N - 1][0]
    return sum


def evaluate(p):
    return 1 / (1 + cost(p))


def select(temp_pop):
    l = list(temp_pop)
    l.sort(key=lambda p: -evaluate(p))
    return np.array(l)[:n_pop]


def best_value(pop):
    ret = 99999999
    for p in pop:
        c = cost(p)
        if c < ret:
            ret = c
    return ret


if __name__ == "__main__":
    history = []
    history.append(best_value(pop))

    for t in range(numloop):
        # Crossover + mutate
        temp_pop = np.empty((2 * n_pop, N), dtype=np.int32)
        temp_pop[:n_pop] = pop
        for i in range(n_pop, 2 * n_pop, 2):
            p1 = pop[np.random.randint(0, n_pop)]
            p2 = pop[np.random.randint(0, n_pop)]
            if np.random.rand() <= rate_of_mutate:
                temp_pop[i] = mutate(p1)
                temp_pop[i + 1] = mutate(p2)
            else:
                temp_pop[i], temp_pop[i + 1] = crossover(p1, p2)

        selected_individuals = select(temp_pop)
        pop = selected_individuals
        history.append(best_value(pop))

    plt.plot(history)
    plt.show()
    print(pop)
    print("best_value=", best_value(pop))
