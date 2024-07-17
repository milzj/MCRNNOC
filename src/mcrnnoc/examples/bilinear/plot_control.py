from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sys

def plot_control(input_dir, colorbar=1):
    cmap_blue_orange = LinearSegmentedColormap.from_list(name="cmap_BlueOrange",
                                                          colors=["tab:blue", "lightgrey", "tab:orange"],
                                                          N=256)

    n = input_dir.split("_n=")[-1].split(".")[0].split("_")[0]
    n = int(n)

    input_filename = input_dir
    u_vec = np.loadtxt(input_filename + ".txt")

    mesh = UnitSquareMesh(n, n)
    U = FunctionSpace(mesh, "DG", 0)
    u = Function(U)
    u.vector()[:] = u_vec
    p = plot(u, wireframe=False, cmap=cmap_blue_orange)
    if colorbar == 1:
        plt.colorbar(p, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(input_dir + ".pdf")
    plt.savefig(input_dir + ".svg")
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <input_dir> [colorbar]")
        sys.exit(1)

    input_dir = "output/" + sys.argv[1]
    colorbar = 1 if len(sys.argv) < 3 else int(sys.argv[2])

    plot_control(input_dir, colorbar)
