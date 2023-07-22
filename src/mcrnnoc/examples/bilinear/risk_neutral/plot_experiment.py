
# source 


from mcrnnoc.stats import load_dict, compute_random_errors
from mcrnnoc.stats import lsqs_label
from mcrnnoc.stats import figure_style

import warnings
import numpy as np
import itertools


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

    except FileNotFoundError:

        stats = {}

        for rank in range(48):

            _filename = filename + "_mpi_rank=" + str(rank)

            try:
                _stats = load_dict(outdir, _filename)
                stats.update(_stats)

            except FileNotFoundError:
                msg = _filename + " not found. " + "Search for simulation output terminates."
                warnings.warn(msg)
                break

    return stats

def plot_experiment(outdir, ndrop=0, tikhonov=-1):
    """Generate convergence plots.

    Parameters:
    ----------
        outdir : string
            directory of experiment
        ndrop : int (optional)
            number of data points to be dropped for computing
            convergence rates using least squares.

    """

    stats = load_experiment(outdir)

    experiment_name = outdir.split("/")[-1].split("_")
    # remove date
    experiment_name.pop(-1)
    experiment_name = "_".join(experiment_name)

    experiment = load_dict(outdir, experiment_name)
    experiment = experiment[experiment_name]

    #     The number of columns that the legend has.
    ncol = 1

    label_realizations = r"$\widetilde \Psi_n(\bar{u}_{N,n})$"
    label_mean_realizations = r"$\widehat{\mathrm{E}}[\widetilde \Psi_n(\bar{u}_{N,n})]$"

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

    # Find all replications for each experiment
    for e in experiment[('n_vec', 'N_vec')]:
        errors[e] = []

        for r in replications:
            s = stats[r][e]

            errors[e].append(s)


    # Compute statistics
    errors_stats = compute_random_errors(errors)

    # Find "x" values
    x_vec = []
    for e in errors.keys():
        n, N = e
        x_vec.append(e[x_id])

    # Compute convergence rates
    y_vec = [errors_stats[e]["mean"] for e in experiments]

    assert len(x_vec) == len(y_vec)
    if least_squares == "standard" and ndrop >= 0:
        ## least squares
        X = np.ones((len(x_vec[ndrop::]), 2)); X[:, 1] = np.log(x_vec[ndrop::]) # design matrix
        x, residudals, rank, s = np.linalg.lstsq(X, np.log(y_vec[ndrop::]), rcond=None)

        rate = x[1]
        constant = np.exp(x[0])

    elif least_squares == "soft_l1" and ndrop >= 0:
        scale = 1e-4
        gtol=1e-10

        import scipy.optimize

        def fun(x, t, y):
            return np.log(y) - x[0] - np.log(t)*x[1]

        x0 = np.ones(2)
        t_train = x_vec[ndrop::]
        y_train = y_vec[ndrop::]
        res_robust = scipy.optimize.least_squares(fun, x0, loss=least_squares, f_scale=scale, gtol=gtol, ftol=gtol, args=(t_train, y_train))

        rate = res_robust.x[1]
        constant = np.exp(res_robust.x[0])
        print("Status Huber regression={}".format(res_robust.status))


    # Plot
    fig, ax = plt.subplots()
    # Plot legend for fixed variable
    ax.plot([], [], " ", label=empty_label)

    # Plot realizations
    for e in errors.keys():
        Y = errors[e]
        ax.scatter(e[x_id]*np.ones(len(Y)), Y, marker="o", color = "black", s=2, label=label_realizations)

    # Plot mean of realizations
    ax.scatter(x_vec, y_vec, marker="s", color="black", label=label_mean_realizations)

    # Plot least squares fit
    if ndrop >= 0:
        X = x_vec[ndrop::]
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


    if experiment_name.find("Synthetic") != -1:

        ax.text(0.5, 0.5, "Test figure with synthetic data.", transform=ax.transAxes,
            fontsize=20, color='red', alpha=1.0,
            ha='center', va='center', rotation='30')


    if set_ylim == True and experiment_name.find("Dimension_Dependence") != -1:
        Y = []
        for e in errors.keys():
                    Y.append(errors[e])

        ymin = np.min(Y)/10
        ymax = 1.5*np.max(Y)

        ax.set_ylim([ymin, ymax])


    plt.tight_layout()
    plt.savefig(outdir + "/" + outdir.split("/")[-1] + "_tikhonov_{}".format(tikhonov) + ".pdf")
    plt.close()

if __name__ == "__main__":

    import sys

    outdir = sys.argv[1]
    try:
        ndrop = int(sys.argv[2])
        tikhonov = int(sys.argv[3])
    except:
        ndrop = 0
        tikhonov = -1

    plot_experiment(outdir, ndrop=ndrop, tikhonov=tikhonov)
