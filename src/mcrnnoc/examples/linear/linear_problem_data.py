from fenics import *
from dolfin_adjoint import *

class LinearProblemData(object):

    def __init__(self, mpi_comm):

        # Taken from section 5.1.1 in https://doi.org/10.1137/S1052623498343131
        self._yd = yd_expr = Expression('(0.25 < x[0] && x[0] < 0.75 && 0.25 < x[1] && x[1] < 0.75) ? -1.0 : 1.0', \
			                    degree=0, mpi_comm=mpi_comm)
        self._lb = Constant(-1.0)
        self._ub = Constant(1.0)
        self._beta = 0.011

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

