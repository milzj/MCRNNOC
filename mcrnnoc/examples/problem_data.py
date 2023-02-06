from fenics import *
from dolfin_adjoint import *

class ProblemData(object):

    def __init__(self):

        self._yd = Expression("sin(2*pi*x[0])*sin(2*pi*x[1])*exp(2.0*x[0])/2.0", degree = 0)
#        self._f = Constant(1.0)
        self._f = Expression("10.0*sin(2*pi*x[0]-x[1])*cos(2*pi*x[1])", degree = 0)
        self._lb = Constant(-1.0)
        self._ub = Constant(1.0)
        self._g = Constant(1.0)
        self._beta = 0.001

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

