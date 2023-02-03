class OptionsSampler(object):

    def __init__(self):

            a = 100.0
            self._options = {
                "std": 1.0,
                "rv_range": [-a, a]
            }


    @property
    def options(self):
        return self._options
