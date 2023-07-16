import numpy as np
from scipy.stats import truncnorm

from .options_sampler import OptionsSampler

class TruncatedGaussianSampler(object):

    def __init__(self):

        options_sampler = OptionsSampler().options
        std = options_sampler["std"]
        rv_range = options_sampler["rv_range"]
        loc = options_sampler["loc"]

        self._seed = 1
        self.std = std
        self.rv_range = rv_range
        self.loc = loc

        self.num_rvs = 1

    @property
    def seed(self):
        return self._seed

    def bump_seed(self):
        self._seed += 1


    def sample(self, sample_index):

        a, b = self.rv_range
        std = self.std
        loc = self.loc

        self.bump_seed()
        np.random.seed(self.seed)

        a_, b_ = (a - loc) / std, (b - loc) / std
        Z = truncnorm.rvs(a_, b_, loc=loc, scale=std, size=self.num_rvs)

        return Z


if __name__ == "__main__":

    sampler = TruncatedGaussianSampler()

    sample = sampler.sample(4)
    print(sample)
