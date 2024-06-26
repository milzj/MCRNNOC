import numpy as np
from scipy.stats import truncnorm

from mcrnnoc.sampler.options_sampler import OptionsSampler

class TruncatedGaussianSampler(object):

    def __init__(self, replication : int, nsamples : int,
                        nrvs : int, nreplications : int):

        entropy = 0x3034c61a9ae04ff8cb62ab8ec2c4b501
        ss = np.random.SeedSequence(entropy)

        self.child_seeds = ss.spawn(nreplications)
        self.streams = [np.random.default_rng(s) for s in self.child_seeds]

        options_sampler = OptionsSampler().options
        std = options_sampler["std"]
        rv_range = options_sampler["rv_range"]
        loc = options_sampler["loc"]

        self.std = std
        self.rv_range = rv_range
        self.loc = loc

        self._sample(replication, nsamples, nrvs)


    def _sample(self, replication, nsamples, nrvs):

        a, b = self.rv_range
        std = self.std
        loc = self.loc

        streams = self.streams

        a_, b_ = (a - loc) / std, (b - loc) / std
        samples = truncnorm.rvs(a_, b_, loc=loc, scale=std, size=(nsamples, nrvs), random_state=streams[replication])

        self.samples = samples

    def sample(self, sample_index):

        return self.samples[sample_index]


if __name__ == "__main__":

    sampler = TruncatedGaussianSampler(0, 4, 3, 10)

    sample = sampler.sample(0)
    print(sample)
    sample = sampler.sample(1)
    print(sample)

