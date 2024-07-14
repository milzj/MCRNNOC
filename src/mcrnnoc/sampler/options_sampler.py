import numpy as np

class OptionsSampler(object):

    def __init__(self):

            std = np.sqrt(1.25)
            self._options = {
                "std": std,
                "rv_range": [-3*std, 3*std],
                "loc": 0.0
            }

    @property
    def options(self):
        return self._options
