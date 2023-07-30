import numpy as np
from scipy.stats import truncnorm
from scipy.stats import qmc

from mcrnnoc.sampler.options_sampler import OptionsSampler

class TruncatedGaussianSobolSampler(object):

    def __init__(self, N):

        if not ((N & (N-1) == 0) and N != 0):
            raise ValueError("N is not 2**m for some natural number m.")

        options_sampler = OptionsSampler().options
        std = options_sampler["std"]
        rv_range = options_sampler["rv_range"]
        loc = options_sampler["loc"]

        self.rv_range = rv_range
        self.std = std
        self.loc = loc
        self._seed = 1

        self.m = int(np.log2(N))


        self._sample_idx = 0
        self.num_rvs = 1

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        self._sample_idx = 0

    @property
    def num_rvs(self):
        return self._num_rvs

    @num_rvs.setter
    def num_rvs(self, num_rvs):
        self._num_rvs = num_rvs
        self.generate_samples()

    def generate_samples(self):

        d = self.num_rvs
        m = self.m
        seed = self.seed
        std = self.std
        a, b = self.rv_range
        loc = self.loc

        sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
        q = sampler.random_base2(m=m)
        a_, b_ = (a - loc) / std, (b - loc) / std
        self.samples = truncnorm.ppf(q, a_, b_, loc=loc, scale=std)

    def bump_seed(self):
        self._seed += 1
        self._sample_idx += 1

    def sample(self, sample_index):
        """Transformed Sobol' sequence

        Parameters:
        -----------
        d : int, optional
            dimension of sequence (number of random variables)
        m : int, optional
            sample size 2**m (so m should be of moderate size)

        """

        s = self.samples[self._sample_idx]
        self.bump_seed()

        return s


if __name__ == "__main__":

    sampler = TruncatedGaussianSobolSampler(4)

    sample = sampler.sample(4)
    print(sample)

    sampler.num_rvs = 10
    sample = sampler.sample(4)
    print(sample)
