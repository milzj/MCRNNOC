import numpy as np

class SolverOptions(object):

    def __init__(self):

        maxiter = 100
        gtol = 1e-6
        ftol = -np.inf

        self._options = {"maxiter": maxiter, "gtol": gtol, "ftol": ftol}
        self._stepsize = "DemyanovRubinovOptimalStepSize"


    @property
    def options(self):
        "Termination options for conditional gradient method."
        return self._options

    @property
    def stepsize(self):
        "Step size rule for conditional gradient method."
        return sel._stepsize
