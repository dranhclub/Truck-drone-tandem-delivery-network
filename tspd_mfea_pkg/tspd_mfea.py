import numpy as np
from mfea_pkg.mfea import MFEA
from .tspd_idvd import TSPD_Individual
from mfea_pkg.evolution_operators import partially_mapped_crossover


class TSPD_MFEA(MFEA):
    def mutate(self, individual) -> TSPD_Individual:
        order, color = individual.genes
        if np.random.rand() < 0.5:  # Mutate on order
            new_order = mutate_order(order)
            return TSPD_Individual((new_order, color), self.num_task)
        else:  # Mutate on color
            mutate_funcs = [add_green, move_green, add_red, move_red, add_green_and_red, remove_green_and_red]
            for f in np.random.permutation(mutate_funcs):
                new_color = f(color)
                if new_color:
                    return TSPD_Individual((order, new_color), self.num_task)

        # If can't mutate color
        new_order = mutate_order(order)
        return TSPD_Individual((new_order, color), self.num_task)

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
            color = fix_color(gen_color_arr(self.genes_length))
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
    return arr


def fix_color(arr):
    red = "R"
    green = "G"
    black = " "
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


def mutate_order(order):
    r = np.random.randint(1, 3)
    (a, b) = sorted(np.random.choice(len(order), 2, replace=False))
    order = order.copy()
    if r == 1:  # Swap 2 point
        order[[a, b]] = order[[b, a]]
    elif r == 2:  # Flip a segment
        temp = order[a:b + 1]
        order[a:b + 1] = temp[::-1]
    elif r == 3:  # Slide
        ida = [m for m in range(a + 1, b + 1)] + [a]
        order[a:b + 1] = order[ida]
    return order


def move_green(color):
    color = color.copy()
    green_idx = [i for i in range(len(color)) if color[i] == "G"]
    green_id = np.random.choice(green_idx)
    if color[green_id - 1] == " ":  # Move to left
        color[green_id - 1] = "G"
        color[green_id] = " "
        return color
    elif color[green_id + 1] != " ":
        return False
    right_color_id = -1
    for i in range(green_id, len(color)):
        if color[i] != " ":
            right_color_id = i
    color[green_id] = " "
    color[right_color_id - 1] = "G"
    return color


def move_red(color):
    color = color.copy()
    red_idx = [i for i in range(len(color)) if color[i] == "R"]
    red_id = np.random.choice(red_idx)
    if color[red_id - 1] == " ":  # Move to left
        color[red_id - 1] = "R"
        color[red_id] = " "
        return color
    elif color[red_id + 1] != " ":
        return False
    right_color_id = -1
    for i in range(red_id, len(color)):
        if color[i] != " ":
            right_color_id = i
    color[red_id] = " "
    color[right_color_id - 1] = "R"
    return color


def add_green(color):
    color = color.copy()
    flag = []
    last = None
    for i in range(len(color)):
        if color[i] != " ":
            if last is not None and color[last] == "R" and color[i] == "R":
                flag.append((last, i))
            last = i
    if len(flag):
        i = np.random.choice(range(len(flag)))
        a, b = flag[i]
        new_gr_id = np.random.choice(range(a + 1, b))
        color[new_gr_id] = "G"
        return color
    else:
        return False


def add_red(color):
    color = color.copy()
    flag = []
    last = None
    for i in range(len(color)):
        if color[i] != " ":
            if (last is not None) \
                    and (i - last > 1) \
                    and (color[last] == "R" and color[i] == "G" or color[last] == "G" and color[i] == "R"):
                flag.append((last, i))
            last = i

    if len(flag):
        i = np.random.choice(range(len(flag)))
        a, b = flag[i]
        new_red_id = np.random.choice(range(a + 1, b))
        color[new_red_id] = "R"
        return color
    else:
        return False


def add_green_and_red(color):
    candidates = []
    last = None
    for i in range(len(color)):
        if color[i] != " ":
            if last is None:
                if i > 1:
                    a, b = sorted(np.random.choice(range(0, i), 2))
                    new_color = color.copy()
                    new_color[a] = "R"
                    new_color[b] = "G"
                    candidates.append(new_color)
            else:
                if i - last > 3:
                    if color[last] == "R" and color[i] == "G":
                        a, b = sorted(np.random.choice(range(last + 1, i), 2))
                        new_color = color.copy()
                        new_color[a] = "G"
                        new_color[b] = "R"
                        candidates.append(new_color)
                    elif color[last] == "G" and color[i] == "R":
                        a, b = sorted(np.random.choice(range(last + 1, i), 2))
                        new_color = color.copy()
                        new_color[a] = "R"
                        new_color[b] = "G"
                        candidates.append(new_color)
                if i - last > 4 and color[last] == color[i] == "R":
                    a, b, c = sorted(np.random.choice(range(last + 1, i), 3))
                    new_color = color.copy()
                    new_color[a] = "G"
                    new_color[b] = "R"
                    new_color[c] = "G"
                    candidates.append(new_color)

            last = i
    if last < len(color) - 2 and color[last] == "R":
        a, b = sorted(np.random.choice(range(last + 1, len(color)), 2))
        new_color = color.copy()
        new_color[a] = "G"
        new_color[b] = "R"
        candidates.append(new_color)

    if len(candidates):
        return np.random.choice(candidates, 1)
    else:
        return False


def remove_green(color):
    color = color.copy()
    flag = []
    c1 = c2 = c3 = c4 = c5 = None
    i1 = i2 = i3 = i4 = i5 = None

    def push_color(c1, c2, c3, c4, c5, c6, i1, i2, i3, i4, i5, i6):
        c1 = c2
        c2 = c3
        c3 = c4
        c4 = c5
        c5 = c6
        i1 = i2
        i2 = i3
        i3 = i4
        i4 = i5
        i5 = i6

    for i in range(len(color)):
        if color[i] != " ":
            push_color(c1, c2, c3, c4, c5, color[i], i1, i2, i3, i4, i5, i)
            if (c1, c2, c3, c4, c5) == ("G", "R", "G", "R", "G"):
                flag.append(i3)

    if len(flag):
        i = np.random.choice(flag)
        color[i] = " "
        return color
    else:
        return False


def remove_red(color):
    color = color.copy()
    flag = []
    last = None
    for i in range(len(color)):
        if color[last] and color[i] == "R":
            flag += [last, i]
        if color[i] != " ":
            last = i

    if len(flag):
        i = np.random.choice(flag)
        color[i] = " "
        return color
    else:
        return False


def remove_green_and_red(color):
    if len([c for c in color if c != " "]) == 3:
        return " " * len(color)
    candidates = []
    c1 = c2 = c3 = None
    i1 = i2 = i3 = None
    for i in range(len(color)):
        c1 = c2
        c2 = c3
        c3 = color[i]
        i1 = i2
        i2 = i3
        i3 = i
        if (c1, c2, c3) == ("G", "R", "G"):
            new_color = color.copy()
            new_color[i1] = " "
            new_color[i2] = " "
            candidates.append(new_color)
            new_color = color.copy()
            new_color[i2] = " "
            new_color[i3] = " "
            candidates.append(new_color)
        elif (c1, c2, c3) == ("R", "R", "G"):
            new_color = color.copy()
            new_color[i2] = " "
            new_color[i3] = " "
            candidates.append(new_color)
            new_color = color.copy()
            new_color[i1] = " "
            new_color[i3] = " "
            candidates.append(new_color)
        elif (c1, c2, c3) == ("G", "R", "R"):
            new_color = color.copy()
            new_color[i1] = " "
            new_color[i2] = " "
            candidates.append(new_color)
            new_color = color.copy()
            new_color[i1] = " "
            new_color[i3] = " "
            candidates.append(new_color)

    cc = [color[i] for i in range(len(color)) if color[i] != " "]
    ci = [color[i] for i in range(len(color)) if color[i] != " "]
    if (cc[0], cc[1], cc[2], cc[3]) == ("R", "G", "R", "G"):
        new_color = color.copy()
        new_color[ci[0]] = new_color[ci[1]] = " "
        candidates.append(new_color)
    elif (cc[0], cc[1], cc[2], cc[3]) == ("R", "G", "R", "R"):
        new_color = color.copy()
        new_color[ci[0]] = new_color[ci[1]] = new_color[ci[2]] = " "
        candidates.append(new_color)
    if (cc[-4], cc[-3], cc[-2], cc[-1]) == ("G", "R", "G", "R"):
        new_color = color.copy()
        new_color[ci[-1]] = new_color[ci[-2]] = " "
        candidates.append(new_color)
    elif (cc[-4], cc[-3], cc[-2], cc[-1]) == ("R", "R", "G", "R"):
        new_color = color.copy()
        new_color[ci[-1]] = new_color[ci[-2]] = new_color[ci[-3]] = " "
        candidates.append(new_color)

    if len(candidates):
        return np.random.choice(candidates, 1)
    else:
        return False
