import numpy as np
from scipy.stats import truncnorm
from scipy.stats import qmc

from options_sampler import OptionsSampler

class ReferenceTruncatedGaussianSobolSampler(object):

    def __init__(self):

        options_sampler = OptionsSampler().options
        std = options_sampler["std"]
        rv_range = options_sampler["rv_range"]

        self.rv_range = rv_range
        self.std = std

        self._seed = 1

    @property
    def seed(self):
        return self._seed

    def bump_seed(self):
        self._seed += 1



    def sample(self, d=2, m=2):
        """Transformed Sobol' sequence

        Parameters:
        -----------
        d : int, optional
            dimension of sequence (number of random variables)
        m : int, optional
            sample size 2**m (so m should be of moderate size)

        """

        std = self.std
        a, b = self.rv_range
        self.bump_seed()
        seed = self.seed

        sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
        q = sampler.random_base2(m=m)


        s = truncnorm.ppf(q, a/std, b/std, loc=0, scale=std)

        return s



if __name__ == "__main__":

    sampler = ReferenceTruncatedGaussianSobolSampler()

    sample = sampler.sample(2, 3)
    print(sample)

    sample = sampler.sample(2, 3)
    print(sample)
