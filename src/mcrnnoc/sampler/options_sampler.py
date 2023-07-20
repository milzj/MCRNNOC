import numpy as np

class OptionsSampler(object):

    def __init__(self):

            a = 100.0
            self._options = {
                "std": np.sqrt(1.25),
                "rv_range": [-a, a],
                "loc": 0.0 
            }


    @property
    def options(self):
        return self._options
