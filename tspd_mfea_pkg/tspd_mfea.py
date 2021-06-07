import numpy as np
from mfea_pkg.mfea import MFEA
from .tspd_idvd import TSPD_Individual
from mfea_pkg.evolution_operators import partially_mapped_crossover


class TSPD_MFEA(MFEA):
    def mutate(self, individual) -> TSPD_Individual:
        r = np.random.randint(1, 3)
        if r == 1:  # Mutate on order
            r = np.random.randint(1, 2)
            if r == 1:  # Swap 2 point
                pass
            elif r == 2:  # Flip a segment
                pass
        elif r == 2:  # Mutate green
            pass
        elif r == 3:  # Mutate red
            pass

    def crossover(self, individual_1, individual_2) -> (TSPD_Individual, TSPD_Individual):
        # TODO: Need more crossover operator
        # r = np.random.randint(0, 2)
        # if r == 0:
        #     pass
        # elif r == 1:
        #     pass
        # elif r == 2:
        #     pass
        # else:
        #     pass

        p1 = individual_1.genes[0]
        p2 = individual_2.genes[0]
        point1, point2 = np.random.choice(len(p1), 2, replace=False)
        c1, c2 = partially_mapped_crossover(p1, p2, point1, point2)
        child_1 = TSPD_Individual((c1, individual_1.genes[1]), self.num_task)
        child_2 = TSPD_Individual((c2, individual_2.genes[1]), self.num_task)
        return child_1, child_2

    def generate_pop(self):
        pop = []
        for i in range(self.pop_num):
            order = np.random.permutation(self.genes_length)
            color = gen_color_arr(self.genes_length)
            pop.append(TSPD_Individual((order, color), self.num_task))
        return pop


def gen_color_arr(length):
    arr = []
    last = None
    last_green = False
    red = "R"
    green = "G"
    black = " "
    for i in range(length):
        if np.random.rand() < 0.5:
            if last == None:
                arr.append(red)
                last = red
            elif last == green:
                arr.append(red)
                last = red
            else:
                if not last_green:
                    arr.append(green)
                    last = green
                    last_green = True
                else:
                    if np.random.rand() < 0.5:
                        arr.append(green)
                        last = green
                        last_green = True
                    else:
                        arr.append(red)
                        last = red
                        last_green = False
        else:
            arr.append(black)
    count = 0
    for i in arr:
        if i == red or i == green:
            count += 1
    if count < 3:
        return [black] * len(arr)

    for i in range(len(arr) - 1, -1, -1):
        if arr[i] == green:
            arr[i] = black
            for j in range(i - 1, -1, -1):
                if arr[j] == red:
                    for k in range(j - 1, -1, -1):
                        if arr[k] == red:
                            arr[j] = black
                            return arr
                        elif arr[k] == green:
                            return arr
            return arr
        elif arr[i] == red:
            for j in range(i - 1, -1, -1):
                if arr[j] == red:
                    arr[i] = black
                    return arr
                elif arr[j] == green:
                    return arr
    return arr
