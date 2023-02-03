class OptionsRandomField(object):

    def __init__(self):

            self._options = {
                "num_addends": 20,
                "len_scale": 0.1
            }


    @property
    def options(self):
        return self._options
