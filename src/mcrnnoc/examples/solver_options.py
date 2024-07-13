import numpy as np
from fw4pde.stepsize import DemyanovRubinovOptimalStepSize
from fw4pde.stepsize import DunnScalingStepSize
from fw4pde.stepsize import DemyanovRubinovAdaptiveStepSize


class SolverOptions(object):

    def __init__(self):

        maxiter = 100
        gtol = 1e-12
        ftol = -np.inf

        self._options = {"maxiter": maxiter, "gtol": gtol, "ftol": ftol}
        self._stepsize = DunnScalingStepSize()
        self._stepsize = DemyanovRubinovAdaptiveStepSize()
        self._stepsize = DemyanovRubinovOptimalStepSize()

    @property
    def options(self):
        "Termination options for conditional gradient method."
        return self._options

    @property
    def stepsize(self):
        "Step size rule for conditional gradient method."
        return self._stepsize
