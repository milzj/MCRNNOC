import pytest

import numpy as np

from fenics import *

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


def test_crit_measure_dual_gap():
    """Compares canonical criticality measure and dual gap

    We check whether the inequality

    canonical criticality measure <= sqrt( dual gap )

    is true. This inequality comes out of some short
    calculations; see
    Lan, https://doi.org/10.1007/978-3-030-39568-1, p. 478

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

    solution = Function(U)
    solution.vector()[:] = -1*np.sign(gradient_vec)

    control = Function(U)
    control.vector()[:] = np.clip(np.random.uniform(U.dim()), lb, ub)

    control_minus_solution = Function(U)
    control_minus_solution_vec = control.vector()[:]-solution.vector()[:]
    control_minus_solution.vector()[:] = control_minus_solution_vec

    dual_gap = assemble(inner(gradient, control_minus_solution)*dx)

    assert criticality_measure(control, gradient, lb, ub, beta) <= np.sqrt(dual_gap)
