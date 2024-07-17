
from mcrnnoc.stats import load_dict, compute_random_errors
from mcrnnoc.stats import lsqs_label
#from mcrnnoc.stats import figure_style

import warnings
import numpy as np
import itertools

import fw4pde
import fenics

from matplotlib import pyplot as plt

plt.rcParams.update({"legend.frameon": True, "legend.loc": "lower left"})
plt.rcParams.update({"legend.columnspacing": 1.0})

def load_experiment(outdir):

    # date of simulation
    filename = outdir
    filename = outdir.split("_")
    filename = filename[-1]

    try:
        stats = load_dict(outdir, filename)
        stats_solutions = load_dict(outdir, filename + "_solutions")

    except FileNotFoundError:

        stats = {}
        stats_solutions = {}

        for rank in range(100):

            _filename = filename + "_mpi_rank=" + str(rank)
            _filename_solutions = filename + "_solutions_mpi_rank=" + str(rank)

            try:
                _stats = load_dict(outdir, _filename)
                stats.update(_stats)

                _stats_solutions = load_dict(outdir, _filename_solutions)
                stats_solutions.update(_stats_solutions)


            except FileNotFoundError:
                msg = _filename + " not found. " + "Search for simulation output terminates."
                warnings.warn(msg)
                break

    return stats, stats_solutions


def plot_data(x_vec, Y_vec, xlabel, label, filename_postfix, base, lsqs_base, empty_label="", ndrop=0):

    ncol = 1
    y_vec = np.mean(Y_vec, axis=1)
    assert len(x_vec) == len(y_vec)

    ## least squares
    X = np.ones((len(x_vec[ndrop::]), 2)); X[:, 1] = np.log(x_vec[ndrop::]) # design matrix
    x, residudals, rank, s = np.linalg.lstsq(X, np.log(y_vec[ndrop::]), rcond=None)

#   X = np.ones((len(x_vec[1:8]), 2)); X[:, 1] = np.log(x_vec[1:8]) # design matrix
#   x, residudals, rank, s = np.linalg.lstsq(X, np.log(y_vec[1:8]), rcond=None)

    rate = x[1]
    constant = np.exp(x[0])

    # Plot
    fig, ax = plt.subplots()
    # Plot legend for fixed variable
    ax.plot([], [], " ", label=empty_label)

    # Plot realizations
    for i in range(len(x_vec)):
        ax.scatter(x_vec[i]*np.ones(len(Y_vec[i])), Y_vec[i], marker="o", color = "black", s=2, label=label)

    # Plot mean of realizations
    ax.scatter(x_vec, y_vec, marker="s", color="black", label="mean")

    # Plot least squares fit
    if ndrop >= 0:
        X = x_vec
        Y = constant*X**rate
        ax.plot(X, Y, color="black", linestyle="--", label=lsqs_label(rate=rate, constant=constant, base=lsqs_base))

    # Legend and labels
    ax.set_xlabel(xlabel)
    ax.set_xscale("log", base=base)
    ax.set_yscale("log", base=base)

    ## Legend with unique entries
    _handles, _labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(_labels, _handles))
    plt.legend(by_label.values(), by_label.keys(), ncol=ncol, loc="best")

    plt.tight_layout()
    plt.savefig(outdir + "/" + outdir.split("/")[-1] + "_{}".format(filename_postfix) + ".svg")
    plt.savefig(outdir + "/" + outdir.split("/")[-1] + "_{}".format(filename_postfix) + ".pdf")
    plt.close()


def plot_experiment(outdir, outdir_ref = "", ndrop=0):
    """Generate convergence plots.

    Parameters:
    ----------
        outdir : string
            directory of experiment
        ndrop : int (optional)
            number of data points to be dropped for computing
            convergence rates using least squares.

    """
    stats, stats_solutions = load_experiment(outdir)
    experiment_name = outdir.split("/")[-1].split("_")
    # remove date
    experiment_name.pop(-1)
    experiment_name = "_".join(experiment_name)

    experiment = load_dict(outdir, experiment_name)
    experiment = experiment[experiment_name]

    # The number of columns that the legend has.
    ncol = 1
    if experiment_name.find("Monte_Carlo_Rate") != -1:
        x_id = 1 # N_vec
        xlabel = r"$N$"
        base = 2
        lsqs_base = "N"
        n = experiment[('n_vec', 'N_vec')][0][0]
        empty_label = r"($n={}$)".format(n)
        set_ylim = False
        ndelete = 0
        least_squares = "standard"
    else:
        raise ValueError(experiment_name + "unknown.")


    experiments = experiment[('n_vec', 'N_vec')]
    replications = sorted(stats.keys())
    errors = {}
    errors_solutions = {}

    label_realizations_vec = [r"$\Psi_{\mathrm{ref}}(\bar{u}_{N})$"]
    label_realizations_vec += [r"$\Psi_{\mathrm{reg},\mathrm{ref}}(\bar{u}_{N})$"]
    label_realizations_vec += [r"$\chi_{\mathrm{ref}}(\bar{u}_{N})$"]

    filename_postfix_vec = ["gap", "regularizedgap", "canonical"]


    # L1 control errors
    if len(outdir_ref) > 0:
        mesh = fenics.UnitSquareMesh(n,n)
        U = fenics.FunctionSpace(mesh, "DG", 0)
        u_minus_uref = fenics.Function(U)
        L1norm = fw4pde.base.NormL1(U)

        filename = np.loadtxt(outdir_ref + "/" + "Reference_Simulation_filename.txt", dtype=str)
        filename = str(filename)
        uref_vec = np.loadtxt("output/"+ filename + ".txt")
        n = int(filename.split("n=")[-1])

    # criticality measures
    for i in range(3):

        label_realizations = label_realizations_vec[i]
        filename_postfix = filename_postfix_vec[i]

        # Find all replications for each experiment
        for e in experiment[('n_vec', 'N_vec')]:
            errors[e] = []
            errors_solutions[e] = []

            for r in replications:
                try:
                    s = stats[r][e][i]
                except:
                    s = stats[r][e]

                errors[e].append(s)

                if i == 0 and len(outdir_ref) > 0:
                    s_solutions = stats_solutions[r][e]
                    u_minus_uref.vector().set_local(s_solutions-uref_vec)
                    errors_solutions[e].append(L1norm(u_minus_uref))

        # Find "x" values
        x_vec = []
        for e in errors.keys():
            n, N = e
            x_vec.append(e[x_id])

        # Compute convergence rates
        y_vec = [errors[e] for e in experiments]

        plot_data(x_vec, y_vec, xlabel, label_realizations, "criticality_measure_" + filename_postfix, base,
                                lsqs_base, empty_label=empty_label, ndrop=ndrop)

        if i == 0 and len(outdir_ref) > 0:
            # Compute convergence rates
            y_vec = [errors_solutions[e] for e in experiments]

            label_realizations = r"$\|\bar u_N - \bar u\|_{L^1(D)}$"
            filename_postfix = "L1errors"
            plot_data(x_vec, y_vec, xlabel, label_realizations, filename_postfix, base,
                                    lsqs_base, empty_label=empty_label, ndrop=ndrop)



if __name__ == "__main__":

    import sys

    outdir = sys.argv[1]

    try:
        outdir_ref = sys.argv[2]
    except:
        outdir_ref = ""

    plot_experiment(outdir, outdir_ref = outdir_ref, ndrop=0)
