# l = 1, a = 1/2, number of odd roots 50

import pytest
import numpy as np
from fenics import *

from mcrnnoc.random_field import ExpRandomField
from .roots import roots

class OptionsRandomField(object):

    def __init__(self):

            self._options = {
                "num_addends": 50,
                "len_scale": 1.0
            }


    @property
    def options(self):
        return self._options


def test_random_field_roots():
    """We compare the roots computed by Python with those
    computed by Mathematica.

    Note: The numbers computed by ExpRandomField need to be
    multiplied by np.pi in order to obtain the roots.

    """

    omega_odd, omega_even = roots()

    mesh = UnitSquareMesh(8,8)
    function_space = FunctionSpace(mesh, "DG", 0)

    random_field_options = OptionsRandomField().options
    n = random_field_options["num_addends"]

    random_field = ExpRandomField(function_space, random_field_options)

    assert len(random_field.odd_roots) +  len(random_field.even_roots) == n

    rtol = 1e-14
    atol = 1e-8

    assert all([np.isclose(np.pi*a,b,rtol=rtol,atol=atol) for a, b in zip(random_field.odd_roots, omega_odd)])
    assert all([np.isclose(np.pi*a,b,rtol=rtol,atol=atol) for a, b in zip(random_field.even_roots, omega_even)])

