
import numpy as np

from fenics import *
import pytest

from mcrnnoc.misc.criticality_measure import criticality_measure

def test_criticality_measure():
    """Testing criticality measure  on problem with known solution.

    For a given function g, and bounds lb, ub, we consider

    min (g, u) s.t. u in [lb, ub].
    """

    n = 8
    tol = 1e-16

    mesh = UnitSquareMesh(n,n)
    U = FunctionSpace(mesh, "DG", 0)

    lb = -1.0
    ub = 1.0
    beta = .0

    gradient = Function(U)
    gradient_vec = np.random.randn(U.dim())
    gradient.vector()[:] = gradient_vec

    control = Function(U)
    control.vector()[:] = -1*np.sign(gradient_vec)

    assert criticality_measure(control, gradient, lb, ub, beta) <= tol
