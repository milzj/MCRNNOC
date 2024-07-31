from fenics import *
from dolfin_adjoint import *

class ProblemData(object):

    def __init__(self, mpi_comm):

        # Taken from section 5.1.1 in https://doi.org/10.1137/S1052623498343131
        self._yd = Expression("sin(2*pi*x[0])*sin(2*pi*x[1])*exp(2.0*x[0])/6.0", degree = 0, mpi_comm=mpi_comm)
        self._f = Expression("10.0*sin(2*pi*x[0]-x[1])*cos(2*pi*x[1])", degree = 0, mpi_comm=mpi_comm)
        self._lb = Constant(0.0)
        self._ub = Constant(1.0)
        self._g = Constant(1.0)
        self._beta = 0.0005

    @property
    def yd(self):
        "Desired state."
        return self._yd

    @property
    def f(self):
        "Right-hand side."
        return self._f

    @property
    def g(self):
        return self._g

    @property
    def lb(self):
        "Lower bound (control)."
        return self._lb

    @property
    def ub(self):
        "Upper bound (control)."
        return self._ub

    @property
    def beta(self):
        "Sparsity parameter."
        return self._beta

