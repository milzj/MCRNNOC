import numpy as np
from scipy.stats import truncnorm
from scipy.stats import qmc

from .options_sampler import OptionsSampler

class ReferenceTruncatedGaussianSampler(object):

    def __init__(self, num_rvs=3, Nref=4):

        if not ((Nref & (Nref-1) == 0) and Nref != 0):
            raise ValueError("Nref is not 2**m for some natural number m.")

        options_sampler = OptionsSampler().options
        std = options_sampler["std"]
        rv_range = options_sampler["rv_range"]
        loc = options_sampler["loc"]

        self.rv_range = rv_range
        self.std = std
        self.loc = loc

        m = int(np.log2(Nref))

        self.samples = self.reference_samples(d=num_rvs, m=m)


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
        loc = self.loc

        sampler = qmc.Sobol(d=d, scramble=False)
        q = sampler.random_base2(m=m)
        q = q + 1.0/(2*2**m)

        assert np.all(q < 1.0), "Invalid shift of Sobol' sequence."

        a_, b_ = (a - loc) / std, (b - loc) / std
        s = truncnorm.ppf(q, a_, b_, loc=loc, scale=std)

        return s

    def sample(self, sample_index):
        """Generates 'samples' from a discrete distribution."""
        return self.samples[sample_index]

if __name__ == "__main__":

    sampler = ReferenceTruncatedGaussianSampler(num_rvs=3, Nref=4)
    print(sampler.sample(0))
    print(sampler.sample(1))
    print(sampler.sample(2))
    print(sampler.sample(3))
