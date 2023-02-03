import numpy as np
from scipy.stats import truncnorm
from scipy.stats import qmc

from options_sampler import OptionsSampler

class ReferenceTruncatedGaussianSampler(object):

    def __init__(self):

        options_sampler = OptionsSampler().options
        std = options_sampler["std"]
        rv_range = options_sampler["rv_range"]

        self.rv_range = rv_range
        self.std = std

    def reference_samples(self, d=2, m=2):
        """Reference sample

        A shifted unscrambled Sobol' sequence is used
        to compute a sample from the truncated normal
        distribution. The shift is taken from p. 73
        in [Owen (2022), https://doi.org/10.1007/978-3-030-98319-2_4]

        Parameters:
        -----------
        d : int, optional
            dimension of sequence (number of random variables)
        m : int, optional
            sample size 2**m (so m should be of moderate size)

        """

        std = self.std
        a, b = self.rv_range

        sampler = qmc.Sobol(d=d, scramble=False)
        q = sampler.random_base2(m=m)
        q = q + 1.0/(2*2**m)

        assert np.all(q < 1.0), "Invalid shift of Sobol' sequence."

        s = truncnorm.ppf(q, a/std, b/std, loc=0, scale=std)

        return s



if __name__ == "__main__":

    sampler = ReferenceTruncatedGaussianSampler()

    sample = sampler.reference_samples(2, 3)
    print(sample)
