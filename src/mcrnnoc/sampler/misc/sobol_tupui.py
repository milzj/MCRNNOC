"""Integration convergence using Sobol' sequence: removing the first point.

Compute the convergence rate for integrating functions using Sobol' low
discrepancy sequence [1]_. We are interested in measuring the effect of
removing the first point of the sequence ([0, ...]).

Two sets of functions are considered:

(i) The first set of functions are synthetic examples specifically designed
to verify the correctness of the implementation [3]_.

(ii) The second set is categorized into types A, B and C [2]_. These categories
state how the variables are important with respect to the function output:

- type A, Functions with a low number of important variables,
- type B, Functions with almost equally important variables but with
  low interactions with each other,
- type C, Functions with almost equally important variables and with
  high interactions with each other.

The theoretical integral for these functions in the unit hypercube is 1.

Quality of the integration is computed using the Root Mean Square Error (RMSE).

.. note:: This script relies on Scipy >= 1.7. Pull Request:
          https://github.com/scipy/scipy/pull/10844

References
----------

.. [1] I. M. Sobol. The distribution of points in a cube and the accurate
   evaluation of integrals. Zh. Vychisl. Mat. i Mat. Phys., 7:784-802,
   1967.

.. [2] Sergei Kucherenko and Daniel Albrecht and Andrea Saltelli. Exploring
   multi-dimensional spaces: a Comparison of Latin Hypercube and Quasi Monte
   Carlo Sampling Techniques. arXiv 1505.02350, 2015.

.. [3] Art B. Owen. On dropping the first Sobol' point. arXiv 2008.08051,
   2020.

---------------------------

MIT License

Copyright (c) 2020 Pamphile Tupui ROY

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
import os
from collections import namedtuple

import numpy as np
from scipy.stats import qmc
from scipy import stats
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from matplotlib.container import ErrorbarContainer
from matplotlib.legend_handler import HandlerErrorbar

path = 'sobol_convergence'
os.makedirs(path, exist_ok=True)
generate = True
n_conv = 999
ns_gen = 2 ** np.arange(4, 13)  # 13

# Functions definitions
_exp1 = 1 - np.exp(1)


def art_1(sample):
    # dim 5, true value 0
    return np.sum(np.exp(sample) + _exp1, axis=1)


def art_2(sample):
    # dim 3, true value 5/3 + 5*(5 - 1)/4
    return np.sum(sample, axis=1) ** 2


def art_3(sample):
    # dim 3, true value 0
    return np.prod(np.exp(sample) + _exp1, axis=1)


def type_a(sample, dim=30):
    # true value 1
    a = np.arange(1, dim + 1)
    f = 1.
    for i in range(dim):
        f *= (abs(4. * sample[:, i] - 2) + a[i]) / (1. + a[i])
    return f


def type_b(sample, dim=30):
    # true value 1
    f = 1.
    for d in range(1, dim + 1):
        f *= (d - sample[:, d - 1]) / (d - 0.5)
    return f


def type_c(sample, dim=10):
    # true value 1
    f = 2 ** dim * np.prod(sample, axis=1)
    return f


functions = namedtuple('functions', ['name', 'func', 'dim', 'ref'])

benchmark = [
    functions('Art 1', art_1, 5, 0),
    functions('Art 2', art_2, 5, 5 / 3 + 5 * (5 - 1) / 4),
    functions('Art 3', art_3, 3, 0),
    functions('Type A', type_a, 30, 1),
    functions('Type B', type_b, 30, 1),
    functions('Type C', type_c, 10, 1)
]


def conv_method(sampler, func, n_samples, n_conv, ref):
    samples = [sampler(n_samples) for _ in range(n_conv)]
    samples = np.array(samples)

    evals = [np.sum(func(sample)) / n_samples for sample in samples]
    squared_errors = (ref - np.array(evals)) ** 2
    rmse = (np.sum(squared_errors) / n_conv) ** 0.5

    if n_conv > 1:
        ci = np.sqrt(stats.t.interval(0.95, len(squared_errors) - 1,
                                      loc=squared_errors.mean(),
                                      scale=stats.sem(squared_errors)))
        ci = [rmse - ci[0], ci[1] - rmse]
    else:
        ci = [0, 0]

    #c1, c2 = stats.chi2.ppf([0.025, 1 - 0.025], n_conv)
    #ci = [rmse * (1 - np.sqrt(n_conv/c2)), rmse * (np.sqrt(n_conv/c1) - 1)]

    return rmse, ci[0], ci[1]  # 2 * np.std(evals) / np.sqrt(n_conv)


# Analysis
if generate:
    sample_mc_rmse = []
    sample_sobol_0_rmse = []
    sample_sobol_no_0_rmse = []
    sample_sobol_scramble_0_rmse = []
    sample_sobol_scramble_no_0_rmse = []

    for ns in ns_gen:
        print(f'-> ns={ns}')

        _sample_mc_rmse = []
        _sample_sobol_0_rmse = []
        _sample_sobol_no_0_rmse = []
        _sample_sobol_scramble_0_rmse = []
        _sample_sobol_scramble_no_0_rmse = []
        for case in benchmark:
            # Monte Carlo
            sampler_mc = lambda x: np.random.random((x, case.dim))
            conv_res = conv_method(sampler_mc, case.func, ns, n_conv, case.ref)
            _sample_mc_rmse.append(conv_res)

            # Sobol' with zero
            engine = qmc.Sobol(d=case.dim, scramble=False)
            conv_res = conv_method(engine.random, case.func, ns, 1, case.ref)
            _sample_sobol_0_rmse.append(conv_res)

            # Sobol' without zero
            def _sampler_sobol_no_0(ns):
                engine = qmc.Sobol(d=case.dim, scramble=False)
                return engine.random(ns + 1)[1:]
            conv_res = conv_method(_sampler_sobol_no_0, case.func, ns - 1, 1, case.ref)
            _sample_sobol_no_0_rmse.append(conv_res)

            # Sobol' scrambled with zero
            def _sampler_sobol_scrambled_0(ns):
                engine = qmc.Sobol(d=case.dim, scramble=True)
                return engine.random(ns)
            conv_res = conv_method(_sampler_sobol_scrambled_0, case.func, ns, n_conv, case.ref)
            _sample_sobol_scramble_0_rmse.append(conv_res)

            # Sobol' scrambled without zero
            def _sampler_sobol_scrambled_no_0(ns):
                engine = qmc.Sobol(d=case.dim, scramble=True)
                return engine.random(ns + 1)[1:]
            conv_res = conv_method(_sampler_sobol_scrambled_no_0, case.func, ns - 1, n_conv, case.ref)
            _sample_sobol_scramble_no_0_rmse.append(conv_res)

        sample_mc_rmse.append(_sample_mc_rmse)
        sample_sobol_0_rmse.append(_sample_sobol_0_rmse)
        sample_sobol_no_0_rmse.append(_sample_sobol_no_0_rmse)
        sample_sobol_scramble_0_rmse.append(_sample_sobol_scramble_0_rmse)
        sample_sobol_scramble_no_0_rmse.append(_sample_sobol_scramble_no_0_rmse)

    np.save(os.path.join(path, 'mc.npy'), sample_mc_rmse)
    np.save(os.path.join(path, 'sobol_0.npy'), sample_sobol_0_rmse)
    np.save(os.path.join(path, 'sobol_no_0.npy'), sample_sobol_no_0_rmse)
    np.save(os.path.join(path, 'sobol_scramble_0.npy'), sample_sobol_scramble_0_rmse)
    np.save(os.path.join(path, 'sobol_scramble_no_0.npy'), sample_sobol_scramble_no_0_rmse)
else:
    sample_mc_rmse = np.load(os.path.join(path, 'mc.npy'))
    sample_sobol_0_rmse = np.load(os.path.join(path, 'sobol_0.npy'))
    sample_sobol_no_0_rmse = np.load(os.path.join(path, 'sobol_no_0.npy'))
    sample_sobol_scramble_0_rmse = np.load(os.path.join(path, 'sobol_scramble_0.npy'))
    sample_sobol_scramble_no_0_rmse = np.load(os.path.join(path, 'sobol_scramble_no_0.npy'))

sample_mc_rmse = np.array(sample_mc_rmse)
sample_sobol_0_rmse = np.array(sample_sobol_0_rmse)
sample_sobol_no_0_rmse = np.array(sample_sobol_no_0_rmse)
sample_sobol_scramble_0_rmse = np.array(sample_sobol_scramble_0_rmse)
sample_sobol_scramble_no_0_rmse = np.array(sample_sobol_scramble_no_0_rmse)

# Plot
for i, case in enumerate(benchmark):
    func = case.name
    fig, ax = plt.subplots()

    ratio_1 = sample_sobol_0_rmse[:, i, 0][0] / ns_gen[0] ** (-2/2)
    # ratio_1 = sample_sobol_scramble_rmse[:, i, 0][0] / ns_gen[0] ** (-2 / 2)
    # ratio_2 = sample_sobol_scramble_0_rmse[:, i, 0][0] / (np.log2(ns_gen[0]) * ns_gen[0] ** (-3 / 2))
    ratio_3 = sample_sobol_scramble_0_rmse[:, i, 0][0] / (ns_gen[0] ** (-3 / 2))

    # ax.plot(ns_gen, ns_gen ** (-1 / 2), ls='-', c='k')
    ax.plot(ns_gen, ns_gen ** (-2/2) * ratio_1, ls='-', c='k')
    # ax.plot(ns_gen, np.log2(ns_gen) * ns_gen ** (-3 / 2) * ratio_2, ls='-.')
    ax.plot(ns_gen, ns_gen ** (-3 / 2) * ratio_3, ls='-', c='k')

    #ax.errorbar(ns_gen, sample_mc_rmse[:, i, 0], sample_mc_rmse[:, i, 1],
    #            ls='None', marker='x', label="MC", c='k')
    ax.plot(ns_gen, sample_sobol_no_0_rmse[:, i, 0],
            ls='None', marker='s', label="Sobol' no 0", c='k')
    ax.plot(ns_gen, sample_sobol_0_rmse[:, i, 0],
            ls='None', marker='o', label="Sobol' with 0", c='k')
    ax.errorbar(ns_gen, sample_sobol_scramble_no_0_rmse[:, i, 0],
                yerr=sample_sobol_scramble_no_0_rmse[:, i, 1:3].T.reshape(2, -1),
                ls='None', marker='+', label="Sobol' scrambled no 0", c='k')
    ax.errorbar(ns_gen, sample_sobol_scramble_0_rmse[:, i, 0],
                yerr=sample_sobol_scramble_0_rmse[:, i, 1:3].T.reshape(2, -1),
                ls='None', marker='^', label="Sobol' scrambled with 0", c='k')

    ax.set_xlabel(r'$N_s$')

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xticks(ns_gen)
    ax.set_xticklabels([fr'$2^{{{ns}}}$' for ns in np.arange(4, 20)])
    ax.set_ylabel(r'$\epsilon$')
    fig.legend(labelspacing=0.7, bbox_to_anchor=(0.5, 0.43),
               handler_map={ErrorbarContainer: HandlerErrorbar(xerr_size=0.7)})
    fig.tight_layout()
    #plt.show()
    fig.savefig(os.path.join(path, f'sobol_conv_integration_{func}.pdf'),
                transparent=True, bbox_inches='tight')
    
