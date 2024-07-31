import numpy as np
from scipy.stats import truncnorm
from scipy.stats import qmc

from mcrnnoc.sampler.options_sampler import OptionsSampler

class ReferenceTruncatedGaussianSampler(OptionsSampler):

    def __init__(self, num_rvs=3, Nref=4, scramble=False):

        if not ((Nref & (Nref-1) == 0) and Nref != 0):
            raise ValueError("Nref is not 2**m for some natural number m.")

        super().__init__()

        self._streams()

        self.std = self.options["std"]
        self.rv_range = self.options["rv_range"]
        self.loc = self.options["loc"]

        a, b = self.rv_range
        std = self.std
        loc = self.loc

        a_, b_ = (a - loc) / std, (b - loc) / std
        self._a = a_
        self._b = b_

        self.scramble = scramble

        m = int(np.log2(Nref))

        self.samples = self.reference_samples(d=num_rvs, m=m)

    def _streams(self, nreplications=1):

        self.entropy = 0x3034c61a9ae04ff8cb62ab8ec2c4b501
        self.shift = 1000
        self.ss = np.random.SeedSequence(self.entropy)
        self.child_seeds = self.ss.spawn(self.shift+nreplications)
        self.streams = [np.random.default_rng(s) for s in self.child_seeds]


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
        loc = self.loc
        a_ = self._a
        b_ = self._b

        if self.scramble == False:
            sampler = qmc.Sobol(d=d, scramble=False)
            q = sampler.random_base2(m=m)

        elif self.scramble == True:
            stream = self.streams[0]
            sampler = qmc.Sobol(d=d, scramble=True, seed=stream)
            q = sampler.random_base2(m=m)

        elif self.scramble == "Uniform":
            stream = self.streams[0]
            q = stream.uniform(0.0, 1.0, (2**m, d))

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

    sampler = ReferenceTruncatedGaussianSampler(num_rvs=3, Nref=4, scramble=True)
    print(sampler.sample(0))
    print(sampler.sample(1))
    print(sampler.sample(2))
    print(sampler.sample(3))
