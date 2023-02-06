import fenics

from mcrnnoc.sampler import TruncatedGaussianSampler
from exp_random_field import ExpRandomField

import matplotlib.pyplot as plt

import numpy as np

def plot_exp_random_field(outdir, n, num_addends):

    sampler = TruncatedGaussianSampler()
    N = 20

    mesh = fenics.UnitSquareMesh(n,n)
    U = fenics.FunctionSpace(mesh, "CG", 1)
    u = fenics.Function(U)
    exp_kappa = ExpRandomField(U)
    num_rvs = exp_kappa.num_rvs

    for i in range(N):

        sample = sampler.sample(num_rvs)
        exp_kappa_sample = exp_kappa.sample(sample)

        u.interpolate(exp_kappa_sample)

        c = fenics.plot(u)
        plt.colorbar(c)

        plt.title(r"Sample of $\kappa$ $(i={})$".format(i))
        filename = outdir + "exp_kappa" + "_sample=" + str(i)
        plt.savefig(filename)
        plt.close()


if __name__ == "__main__":

    import sys, os

    n = 128
    num_addends = 20

    outdir = "exp_random_field/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)


    plot_exp_random_field(outdir, n, num_addends)
