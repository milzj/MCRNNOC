from fenics import *
from dolfin_adjoint import *

class LinearProblemData(object):

    def __init__(self, mpi_comm):

        # Taken from section 5.1.1 in https://doi.org/10.1137/S1052623498343131
        self._yd = Expression("sin(2*pi*x[0])*sin(2*pi*x[1])*exp(2.0*x[0])/6.0", degree = 0, mpi_comm=mpi_comm)
        self._lb = Constant(-2.0)
        self._ub = Constant(2.0)
        self._beta = 0.005

    @property
    def yd(self):
        "Desired state."
        return self._yd

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

