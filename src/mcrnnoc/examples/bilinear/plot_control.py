from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from mcrnnoc.stats import figure_style

import sys

input_dir = "output/" + sys.argv[1]
n = input_dir.split("_n=")[-1].split(".")[0].split("_")[0]
n = int(n)

input_filename = input_dir
u_vec = np.loadtxt(input_filename + ".txt")

mesh = UnitSquareMesh(n, n)
U = FunctionSpace(mesh, "DG", 0)
u = Function(U)
u.vector()[:] = u_vec

p = plot(u, wireframe=False, cmap=cm.coolwarm)
# https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
plt.colorbar(p, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(input_dir + ".pdf")
