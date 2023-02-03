import numpy as np

class OptionsSampler(object):

    def __init__(self):

            a = 100.0
            self._options = {
                "std": np.sqrt(2.0),
                "rv_range": [-a, a]
            }


    @property
    def options(self):
        return self._options
