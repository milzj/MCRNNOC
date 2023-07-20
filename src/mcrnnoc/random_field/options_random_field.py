class OptionsRandomField(object):

    def __init__(self):

            self._options = {
                "num_addends": 10,
                "len_scale": 1.0
            }


    @property
    def options(self):
        return self._options
