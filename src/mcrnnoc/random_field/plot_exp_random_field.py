import fenics

from mcrnnoc.sampler import TruncatedGaussianSampler
from mcrnnoc.sampler import ReferenceTruncatedGaussianSampler
from exp_random_field import ExpRandomField

from options_random_field import OptionsRandomField

import matplotlib.pyplot as plt

import numpy as np

def plot_exp_random_field(outdir, n, num_addends):

    N = 2**14

    mesh = fenics.UnitSquareMesh(n,n)
    U = fenics.FunctionSpace(mesh, "DG", 0)
    options_random_field = OptionsRandomField()
    exp_kappa = ExpRandomField(U, options_random_field.options)
    num_rvs = exp_kappa.num_rvs

    sampler = ReferenceTruncatedGaussianSampler(num_rvs=num_rvs, Nref=N, scramble=False)
    Nplot = min(10, N)

    exp_kappa_samples = []
    umins = []
    umaxs = []
    conds = []

    for i in range(N):
        sample = sampler.sample(i)
        exp_kappa_sample = exp_kappa.sample(sample)

        u = fenics.Function(U)
        u.interpolate(exp_kappa_sample)
        if i <= Nplot:
            exp_kappa_samples.append(u)

        u_vec = u.vector()[:]

        umins.append(min(u_vec))
        umaxs.append(max(u_vec))
        # condition number
        cond = max(u_vec)/min(u_vec)
        conds.append(cond)

    print("========================================")
    print("Characteristics of random field")
    print("options random field={}".format(options_random_field.options))
    print("sampler std={}".format(sampler.std))
    print("minimum of random field={}".format(np.min(umins)))
    print("mean of (1/minimum of random field)={}".format(np.mean([1/umin for umin in umins])))
    print("std of (1/minimum of random field)={}".format(np.std([1/umin for umin in umins])))
    print("maximum of random field={}".format(np.max(umaxs)))
    print("average condition number of random field={}".format(np.mean(conds)))
    print("median condition number of random field={}".format(np.median(conds)))
    print("std condition number of random field={}".format(np.std(conds)))
    print("worst case condition number of random field={}".format(np.max(conds)))
    print("(estimated using N={} samples)".format(N))
    print("========================================")

    for i in range(Nplot):
        u = exp_kappa_samples[i]

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

    outdir = "exp_random_field_plots/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)


    plot_exp_random_field(outdir, n, num_addends)
