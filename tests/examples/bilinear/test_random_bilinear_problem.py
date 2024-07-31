
import pytest
import numpy as np

from mcrnnoc.examples.bilinear.random_bilinear_problem import RandomBilinearProblem


def test_random_bilinear_problem():


    n = 8
    random_problem = RandomBilinearProblem(n)
    z = np.ones(random_problem.num_addends**2)

    assert random_problem(random_problem.u, z) >= 0.0

