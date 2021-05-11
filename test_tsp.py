from unittest import TestCase
from tsp_ga import _crossover
import numpy as np


class Test(TestCase):
    def test_crossover(self):
        p1 = np.array([3, 4, 8, 2, 7, 1, 6, 5])
        p2 = np.array([4, 2, 5, 1, 6, 8, 3, 7])
        desire_child_1 = np.array([3, 4, 2, 1, 6, 8, 7, 5])
        desire_child_2 = np.array([4, 8, 5, 2, 7, 1, 3, 6])
        desire_ret = (desire_child_1, desire_child_2)
        np.testing.assert_array_equal(_crossover(p1, p2, 2, 5), desire_ret)
