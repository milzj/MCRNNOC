import numpy as np
from scipy.stats import truncnorm
from scipy.stats import qmc

from .options_sampler import OptionsSampler

class TruncatedGaussianSobolSampler(object):

    def __init__(self):

        options_sampler = OptionsSampler().options
        std = options_sampler["std"]
        rv_range = options_sampler["rv_range"]
        loc = options_sampler["loc"]

        self.rv_range = rv_range
        self.std = std
        self.loc = loc
        self._seed = 1

        self.num_rvs = 1


    @property
    def seed(self):
        return self._seed

    def bump_seed(self):
        self._seed += 1



    def sample(self, sample_index):
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
        loc = self.loc
        self.bump_seed()
        seed = self.seed

        d = self.num_rvs
        sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
        q = sampler.random_base2(m=0)[0]

        a_, b_ = (a - loc) / std, (b - loc) / std
        s = truncnorm.ppf(q, a_, b_, loc=loc, scale=std)

        return s



if __name__ == "__main__":

    sampler = ReferenceTruncatedGaussianSobolSampler()

    sample = sampler.sample(4)
    print(sample)

    sampler.num_rvs = 10
    sample = sampler.sample(4)
    print(sample)
