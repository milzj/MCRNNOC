import numpy as np

class OptionsSampler(object):

    def __init__(self):

            a = 2.0
            self._options = {
                "std": np.sqrt(a),
                "rv_range": [-3*a, 3*a],
                "loc": 0.0
            }

    @property
    def options(self):
        return self._options
