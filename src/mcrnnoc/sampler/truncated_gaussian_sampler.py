import numpy as np
from scipy.stats import truncnorm

from mcrnnoc.sampler.reference_truncated_gaussian_sampler import ReferenceTruncatedGaussianSampler

class TruncatedGaussianSampler(ReferenceTruncatedGaussianSampler):

    def __init__(self, replication : int, nsamples : int,
                        nrvs : int, nreplications : int):

        super().__init__()

        self._streams(nreplications=nreplications)
        self._sample(replication, nsamples, nrvs)

    def _sample(self, replication, nsamples, nrvs):

        streams = self.streams

        a_, b_ = self._a, self._b
        loc = self.loc
        std = self.std

        samples = truncnorm.rvs(a_, b_, loc=loc, scale=std, size=(nsamples, nrvs),
                    random_state=streams[replication])

        self.samples = samples

    def sample(self, sample_index):

        return self.samples[sample_index]


if __name__ == "__main__":

    sampler = TruncatedGaussianSampler(0, 4, 3, 10)

    sample = sampler.sample(0)
    print(sample)
    sample = sampler.sample(1)
    print(sample)

