import numpy as np


def mutate(p):
    point_1, point_2 = sorted(np.random.choice(len(p), 2, replace=False))
    # Flip
    ret = np.copy(p)
    tmp = p[point_1:point_2 + 1]
    ret[point_1:point_2 + 1] = tmp[::-1]
    return ret


# Partially Mapped Crossover Operator
def partially_mapped_crossover(p1, p2, point_1, point_2):
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
