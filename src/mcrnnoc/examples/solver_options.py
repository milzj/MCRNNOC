import numpy as np
from fw4pde.stepsize import DemyanovRubinovOptimalStepSize

class SolverOptions(object):

    def __init__(self):

        maxiter = 30
        gtol = 1e-8
        ftol = -np.inf

        self._options = {"maxiter": maxiter, "gtol": gtol, "ftol": ftol}
        self._stepsize = DemyanovRubinovOptimalStepSize()


    @property
    def options(self):
        "Termination options for conditional gradient method."
        return self._options

    @property
    def stepsize(self):
        "Step size rule for conditional gradient method."
        return self._stepsize
