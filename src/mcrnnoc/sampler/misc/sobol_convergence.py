"""Sobol' sequence

Compute the convergence rate for integrating functions
using Sobol' sequences. We are interested in measuring
the effect of shifting an unscrambled Sobol' sequence
using the shift discussed in [2]_ on the integration
quality.

We perform simulations for two test functions:

(1) A test function considered in [2]_; here called
art_1.

(2) The test function "type_a" considered in [1]_.

References:
-----------

..  [1] Pamphile Tupui Roy,
    https://gist.github.com/tupui/fb6e219b1dd2316b7498ebce231bfff5, 2020

..  [2] Art B. Owen. On dropping the first Sobol’ point. In A. Keller, editor,
    Monte Carlo and quasi-Monte Carlo methods,
    Springer Proc. Math. Stat. 387, pages 71–86. Springer, Cham, 2022.
    doi:10.1007/978-3-030-98319-2_4.
..  [3] Norton, Khokhlov, and Uryasev, (2019),
    Calculating CVaR and bPOE for common probability distributions with
    application to portfolio optimization and density estimation,
    https://doi.org/10.1007/s10479-019-03373-1
"""

import os
from collections import namedtuple

import numpy as np
from scipy.stats import qmc
from scipy import stats
from scipy.stats import norm, lognorm
from scipy import special

import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsfonts}",
	"font.family": "serif",
	"font.serif": "Computer Modern Roman",
	"font.monospace": "Computer Modern Typewriter",
    "lines.linewidth": 5.75,
    "lines.markersize": 15,
	"font.size": 16.5})

ms = np.arange(10, 16)

path = 'sobol_convergence'
os.makedirs(path, exist_ok=True)

def art_1(sample):
    return np.sum(np.exp(sample) + 1.0 - np.exp(1.0), axis=1)

def type_a(sample, dim=30):
    # true value 1
    a = np.arange(1, dim + 1)
    f = 1.
    for i in range(dim):
        f *= (abs(4. * sample[:, i] - 2) + a[i]) / (1. + a[i])
    return f

def mus_rand_true(c=0.5, p=2):
    # mean upper semideviation of order one
    # https://www.wolframalpha.com/input?i=integrate+max%28x-.5%2C+0%29%5E2+from+x%3D0+to+1
    return .5 + c*0.041666666666666666666**(1/p)


def mus_rand(sample, c=.5, p=2):
    mean_ = np.mean(sample)
    return mean_ + c*np.mean(np.maximum(sample-mean_, 0.0)**p)**(1/p) - mus_rand_true(c=c, p=p)

def mdrm_rand_true(c=0.5):
    # mean upper semideviation of order one
    # https://www.wolframalpha.com/input?i=integrate+max%28x-.5%2C+0%29%5E2+from+x%3D0+to+1
    return c*0.42373366503113304128301257811404833

def mdrm_rand(sample, c=.25):
    sample_ = np.exp(sample)-np.exp(1.0)+1.0
    mean_ = np.mean(sample_)
    return mean_ + c*np.mean(np.abs(sample_-mean_)) - mdrm_rand_true(c=c)

def cvar_gaussian_true(beta):
    # cvar of standard Gaussian
    var = norm.ppf(beta)
    fvar = 1/np.sqrt(2*np.pi)*np.exp(-var**2/2)
    true_var = fvar/(1-beta)
    return true_var

def cvar_lognormal_true(beta, mu=0.0, s=1.0):
    # cvar of log normal
    erfinv_ = special.erfinv(2.0*beta-1.0)
    erf_ = special.erf(s/np.sqrt(2)-erfinv_)
    true_var = 0.5*np.exp(mu+s**2/2.0)*(1+erf_)/(1-beta)

    return true_var

def cvar_gaussian(sample, beta=0.95):
    n = len(sample)
    sample = norm.ppf(sample, loc=0.0, scale=1.0)
    idx = int(np.ceil(n*(1.0-beta)))
    sorted_sample = sorted(sample)[::-1]
    var = sorted_sample[idx-1]
    cvar_emp = var + 1.0/(1.0-beta)*np.mean(np.maximum(sample -var,0.0))
    return cvar_emp - cvar_gaussian_true(beta)

def cvar_lognormal(sample, beta=0.95, s=1.0, loc=0.0, scale=1.0):
    n = len(sample)
    sample = lognorm.ppf(sample, s, loc=loc, scale=scale)
    idx = int(np.ceil(n*(1.0-beta)))
    sorted_sample = sorted(sample)[::-1]
    var = sorted_sample[idx-1]
    cvar_emp = var + 1.0/(1.0-beta)*np.mean(np.maximum(sample -var,0.0))
    return cvar_emp - cvar_lognormal_true(beta, mu=loc, s=1.0)

def conv_method(sampler, func, dim, m, n_conv, mean):

    samples = [sampler(dim, m) for _ in range(n_conv)]
    samples = np.array(samples)

    evals = [np.mean(func(sample)) for sample in samples]
    squared_errors = np.abs(mean - np.array(evals)) **2
    rmse = np.mean(squared_errors) ** 0.5

    return rmse


def _sampler_rand(dim, m):

    return np.random.rand(2**m, dim)


def _sampler_sobol_scrambled(dim, m):

    sampler = qmc.Sobol(d=dim, scramble=True)
    q = sampler.random_base2(m=m)

    return q

def _sampler_halton_scrambled(dim, m):

    sampler = qmc.Halton(d=dim, scramble=True)
    q = sampler.random(2**m)

    return q


def _sampler_sobol_shifted(dim, m):

    sampler = qmc.Sobol(d=dim, scramble=False)
    q = sampler.random_base2(m=m)
    q = q + 1/(2*2**m)

    assert np.all(q < 1.0), "Invalid shift of Sobol' sequence."

    return q

def _sampler_sobol(dim, m):

    sampler = qmc.Sobol(d=dim, scramble=False)
    q = sampler.random_base2(m=m)

    return q


def signif(x, precision=3):
	"""Rounds the input to significant figures.

	Parameters:
	----------
		x : float
			a floating point number

		precision : int (optional)
			number of significant figures

	"""
	y = np.format_float_positional(x, precision=precision, unique=True, trim="k", fractional=False)
	return np.float64(y)

def lsqs_label(constant=0.0, rate=0.0, base=10.0, precision=3):
	constant = signif(constant, precision=precision)
	rate = signif(rate, precision=precision)
	return r"w/ rate ${}\cdot {}^{}$".format(constant, base, "{"+ str(rate)+"}")

def least_squares(x_vec, y_vec, ndrop=0):
    X = np.ones((len(x_vec[ndrop::]), 2)); X[:, 1] = np.log(x_vec[ndrop::]) # design matrix
    x, residudals, rank, s = np.linalg.lstsq(X, np.log(y_vec[ndrop::]), rcond=None)

    rate = x[1]
    constant = np.exp(x[0])
    return rate, constant


functions = namedtuple('functions', ['name', 'func', 'dim', 'mean'])

cases = [
    functions("mean-deviation", mdrm_rand, 1, 0.0),
    functions("Art 1", art_1, 10, 0.0),
    functions("Type A", type_a, 30, 1.0),
    functions("mean-upper-semideviation", mus_rand, 1, 0.0),
    functions("CVaRGauss75", lambda x: cvar_gaussian(x, beta=.75), 1, 0.0),
    functions("CVaRGauss95", lambda x: cvar_gaussian(x, beta=.95), 1, 0.0),
    functions("CVaRGauss99", lambda x: cvar_gaussian(x, beta=.99), 1, 0.0),
    functions("CVaRLogNormal75", lambda x: cvar_lognormal(x, beta=.75), 1, 0.0),
    functions("CVaRLogNormal95", lambda x: cvar_lognormal(x, beta=.95), 1, 0.0),
    functions("CVaRLogNormal99", lambda x: cvar_lognormal(x, beta=.99), 1, 0.0)
]

for case in cases:

    n_conv = 100
    rmse_rand = []
    for m in ms:
        rmse = conv_method(_sampler_rand, case.func, case.dim, m, n_conv, case.mean)
        rmse_rand.append(rmse)
    rate_rand, constant_rand = least_squares(2**ms, rmse_rand, ndrop=0)

    n_conv = 100
    rmse_halton_scrambled = []
    for m in ms:
        rmse = conv_method(_sampler_halton_scrambled, case.func, case.dim, m, n_conv, case.mean)
        rmse_halton_scrambled.append(rmse)
    rate_halton_scrambled, constant_halton_scrambled = least_squares(2**ms, rmse_halton_scrambled, ndrop=0)

    n_conv = 100
    rmse_sobol_scrambled = []
    for m in ms:
        rmse = conv_method(_sampler_sobol_scrambled, case.func, case.dim, m, n_conv, case.mean)
        rmse_sobol_scrambled.append(rmse)
    rate_sobol_scrambled, constant_sobol_scrambled = least_squares(2**ms, rmse_sobol_scrambled, ndrop=0)

    n_conv = 1
    rmse_sobol_shifted = []
    for m in ms:
        rmse = conv_method(_sampler_sobol_shifted, case.func, case.dim, m, n_conv, case.mean)
        rmse_sobol_shifted.append(rmse)
    rate_sobol_shifted, constant_sobol_shifted = least_squares(2**ms, rmse_sobol_shifted, ndrop=0)

    n_conv = 1
    rmse_sobol = []
    for m in ms:
        rmse = conv_method(_sampler_sobol, case.func, case.dim, m, n_conv, case.mean)
        rmse_sobol.append(rmse)
    rate_sobol, constant_sobol = least_squares(2**ms, rmse_sobol, ndrop=0)

    fig, ax = plt.subplots()

    ax.plot(2**ms, rmse_sobol_scrambled, marker="o", linestyle="-", label="scrambled Sobol' " + lsqs_label(rate=rate_sobol_scrambled, constant=constant_sobol_scrambled, base=10) )
    ax.plot(2**ms, rmse_sobol_shifted, marker="s", linestyle="--", label="shifted Sobol' " + lsqs_label(rate=rate_sobol_shifted, constant=constant_sobol_shifted, base=10) )
    ax.plot(2**ms, rmse_sobol, marker="d", linestyle="-.", label="Sobol' " + lsqs_label(rate=rate_sobol, constant=constant_sobol, base=10) )
    ax.plot(2**ms, rmse_halton_scrambled, marker="<", linestyle="-", label="scrambled Halton " + lsqs_label(rate=rate_halton_scrambled, constant=constant_halton_scrambled, base=10) )
    ax.plot(2**ms, rmse_rand, marker="v", linestyle=":", label="Monte Carlo " + lsqs_label(rate=rate_rand, constant=constant_rand, base=10) )

    ax.set_xlabel("samples")
    ax.set_ylabel("root mean square error")
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xticks(2**ms)
    ax.set_xticklabels([fr'$2^{{{m}}}$' for m in ms])

    ax.legend(loc="lower left")
    fig.tight_layout()
    func = case.name
    fig.savefig(os.path.join(path, f"sobol_integration_{func}.png"))
    fig.savefig(os.path.join(path, f"sobol_integration_{func}.pdf"))

