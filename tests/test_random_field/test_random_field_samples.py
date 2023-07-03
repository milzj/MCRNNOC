import pytest
import numpy as np
from fenics import *

from mcrnnoc.random_field import ExpRandomField
from .roots import roots

import itertools

class OptionsRandomField(object):

    def __init__(self):

            self._options = {
                "num_addends": 10,
                "len_scale": 1.0
            }


    @property
    def options(self):
        return self._options


def eigenfunctions(i, k):

    if i % 2 != 0 and k % 2 != 0:
        A_i = "1.0/sqrt(a+sin(2*omega_i*a)/2/omega_i)"
        A_k = "1.0/sqrt(a+sin(2*omega_k*a)/2/omega_k)"
        return A_i+"*"+"cos(omega_i*(x[0]-a))"+"*"+A_k+"*"+"cos(omega_k*(x[1]-a))"
    elif i % 2 != 0 and k % 2 == 0:
        A_i = "1.0/sqrt(a+sin(2*omega_i*a)/2/omega_i)"
        B_k = "1.0/sqrt(a-sin(2*omega_k*a)/2/omega_k)"
        return A_i+"*"+"cos(omega_i*(x[0]-a))"+"*"+B_k+"*"+"sin(omega_k*(x[1]-a))"
    elif i % 2 == 0 and k % 2 != 0:
        B_i = "1.0/sqrt(a-sin(2*omega_i*a)/2/omega_i)"
        A_k = "1.0/sqrt(a+sin(2*omega_k*a)/2/omega_k)"
        return B_i+"*"+"sin(omega_i*(x[0]-a))"+"*"+A_k+"*"+"cos(omega_k*(x[1]-a))"
    elif i % 2 == 0 and k % 2 == 0:
        B_i = "1.0/sqrt(a-sin(2*omega_i*a)/2/omega_i)"
        B_k = "1.0/sqrt(a-sin(2*omega_k*a)/2/omega_k)"
        return B_i+"*"+"sin(omega_i*(x[0]-a))"+"*"+B_k+"*"+"sin(omega_k*(x[1]-a))"

def omegas(i):

    j = int(np.ceil(i/2))

    omega_odd, omega_even = roots()

    if i % 2 != 0:
        return omega_odd[j-1]
    else:
        return omega_even[j-1]

def eigenvalues(l, omega):

    return 2/l/(1/l**2+omega**2)


class ReferenceRandomField(object):

    def __init__(self, function_space, random_field_options):

        self.a = 0.5

        self.function_space = function_space

        self.num_addends = random_field_options["num_addends"]
        self.len_scale = random_field_options["len_scale"]

        self.idx_pairs = list(itertools.product(range(1, self.num_addends+1), repeat=2))


    def __call__(self,z):

        j = 0
        s = np.zeros(self.function_space.dim())
        function_space = self.function_space
        element = function_space.ufl_element()
        len_scale = self.len_scale
        a = self.a

        for (i,k) in self.idx_pairs:
            expression_str = eigenfunctions(i, k)
            omega_i = omegas(i)
            omega_k = omegas(k)
            eigenfunction_ = Expression(expression_str, a =.5, omega_i = omega_i, omega_k = omega_k, element=element)
            eigenfunction = Function(function_space)
            eigenfunction.interpolate(eigenfunction_)
            eigenvalue_i = eigenvalues(len_scale, omega_i)
            eigenvalue_k = eigenvalues(len_scale, omega_k)

            s += z[j]*np.sqrt(eigenvalue_i)*np.sqrt(eigenvalue_k)*eigenfunction.vector()[:]
            j += 1

        rf = Function(function_space)
        rf.vector()[:] = np.exp(s)

        return rf


def test_random_field_samples():

        rtol = 1e-10

        mesh = UnitSquareMesh(64,64)
        function_space = FunctionSpace(mesh, "DG", 0)

        random_field_options = OptionsRandomField().options
        num_addends = random_field_options["num_addends"]

        num_rvs = num_addends**2

        reference_random_field = ReferenceRandomField(function_space, random_field_options)
        random_field = ExpRandomField(function_space, random_field_options)

        for seed in [1234, 234523, 245345]:

            np.random.seed(seed)
            z = 2*np.random.randn(num_rvs)
            reference_rf = reference_random_field(z)
            rf = random_field(z)

            assert errornorm(rf, reference_rf, degree_rise = 0) < rtol
