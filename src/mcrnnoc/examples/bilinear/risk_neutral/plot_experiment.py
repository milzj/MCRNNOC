from mcrnnoc.stats import load_dict, lsqs_label
import warnings
import numpy as np
import fw4pde
import fenics
from matplotlib import pyplot as plt

from mcrnnoc.stats.figure_style import *

plt.rcParams.update({
    "legend.frameon": True,
    "legend.loc": "lower left",
    "legend.columnspacing": 1.0
})

def load_experiment(outdir):
    filename = outdir.split("_")[-1]

    try:
        stats = load_dict(outdir, filename)
        stats_solutions = load_dict(outdir, filename + "_solutions")
    except FileNotFoundError:
        stats, stats_solutions = {}, {}
        for rank in range(100):
            _filename = f"{filename}_mpi_rank={rank}"
            _filename_solutions = f"{filename}_solutions_mpi_rank={rank}"
            try:
                _stats = load_dict(outdir, _filename)
                stats.update(_stats)
                _stats_solutions = load_dict(outdir, _filename_solutions)
                stats_solutions.update(_stats_solutions)
            except FileNotFoundError:
                warnings.warn(f"{_filename} not found. Search for simulation output terminates.")
                break

    return stats, stats_solutions

def plot_data(x_vec, Y_vec, xlabel, label, filename_postfix, base, lsqs_base, empty_label="", ndrop=0, outdir=""):
    y_vec = np.mean(Y_vec, axis=1)
    assert len(x_vec) == len(y_vec)

    X = np.ones((len(x_vec[ndrop:]), 2))
    X[:, 1] = np.log(x_vec[ndrop:])
    x, _, _, _ = np.linalg.lstsq(X, np.log(y_vec[ndrop:]), rcond=None)

    rate, constant = x[1], np.exp(x[0])

    fig, ax = plt.subplots()
    ax.plot([], [], " ", label=empty_label)
    for i in range(len(x_vec)):
        ax.scatter(x_vec[i] * np.ones(len(Y_vec[i])), Y_vec[i], marker="o", color="black", s=2, label=label)
    ax.scatter(x_vec, y_vec, marker="s", color="black", label="mean")

    if ndrop >= 0:
        X_fit, Y_fit = x_vec, constant * x_vec**rate
        ax.plot(X_fit, Y_fit, color="black", linestyle="--", label=lsqs_label(rate=rate, constant=constant, base=lsqs_base))

    ax.set_xlabel(xlabel)
    ax.set_xscale("log", base=base)
    ax.set_yscale("log", base=base)
    _handles, _labels = ax.get_legend_handles_labels()
    by_label = dict(zip(_labels, _handles))
    ax.legend(by_label.values(), by_label.keys(), ncol=1, loc="best")

    plt.tight_layout()
    plot_path = f"{outdir}/{outdir.split('/')[-1]}_{filename_postfix}"
    plt.savefig(f"{plot_path}.svg")
    plt.savefig(f"{plot_path}.pdf")
    plt.close()

def plot_experiment(outdir, outdir_ref="", ndrop=0):
    stats, stats_solutions = load_experiment(outdir)
    experiment_name = "_".join(outdir.split("/")[-1].split("_")[:-1])
    experiment = load_dict(outdir, experiment_name)[experiment_name]

    if "Monte_Carlo_Rate" in experiment_name:
        x_id, xlabel, base, lsqs_base = 1, r"$N$", 2, "N"
        n = experiment[('n_vec', 'N_vec')][0][0]
        empty_label = r"($n={}$)".format(n)
    else:
        raise ValueError(f"{experiment_name} unknown.")

    experiments, replications = experiment[('n_vec', 'N_vec')], sorted(stats.keys())
    errors, errors_solutions, optimality_gap = {}, {}, {}

    labels = [r"$\Psi_{\mathrm{ref}}(u_{N}^*)$", r"$\Psi_{\mathrm{ref}}(u_{N}^*)$", r"$\chi_{\mathrm{ref}}(u_{N}^*)$", r"$r_{\mathcal{G}}(u_N^*)$"]
    postfixes = ["dualgap", "regularizedgap", "canonical", "optimalitygap"]

    if outdir_ref:
        mesh = fenics.UnitSquareMesh(n, n)
        U = fenics.FunctionSpace(mesh, "DG", 0)
        u_minus_uref = fenics.Function(U)
        L1norm = fw4pde.base.NormL1(U)
        ref_filename = np.loadtxt(f"{outdir_ref}/Reference_Simulation_filename.txt", dtype=str)
        uref_vec = np.loadtxt(f"output/{ref_filename}.txt")
        ref_filename = str(ref_filename).replace("solution", "optimal_value")
        ref_optval = np.loadtxt(f"output/{ref_filename}.txt")

    for i, (label_realizations, filename_postfix) in enumerate(zip(labels, postfixes)):
        for e in experiments:
            errors[e], errors_solutions[e] = [], []
            for r in replications:
                try:
                    s = stats[r][e][i]
                except (IndexError, KeyError):
                    s = stats[r][e]
                errors[e].append(s)

                if i == 0 and outdir_ref:
                    s_solutions = stats_solutions[r][e]
                    u_minus_uref.vector().set_local(s_solutions - uref_vec)
                    errors_solutions[e].append(L1norm(u_minus_uref))


        x_vec = [e[x_id] for e in errors.keys()]
        y_vec = [errors[e] for e in experiments]

        if i == 3 and outdir_ref:
            y_vec = [y-ref_optval for y in y_vec]

        plot_data(x_vec, y_vec, xlabel, label_realizations, f"criticality_measure_{filename_postfix}", base, lsqs_base, empty_label=empty_label, ndrop=ndrop, outdir=outdir)

        if i == 0 and outdir_ref:
            y_vec = [errors_solutions[e] for e in experiments]
            plot_data(x_vec, y_vec, xlabel, r"$\|u_N^* - u^*\|_{L^1(D)}$", "L1errors", base, lsqs_base, empty_label=empty_label, ndrop=ndrop, outdir=outdir)

if __name__ == "__main__":
    import sys
    outdir = sys.argv[1]
    outdir_ref = sys.argv[2] if len(sys.argv) > 2 else ""
    plot_experiment(outdir, outdir_ref=outdir_ref, ndrop=0)

