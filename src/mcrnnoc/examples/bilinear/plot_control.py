from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


cmap_blue_orange = LinearSegmentedColormap.from_list(name="cmap_BlueOrange",
                                          colors =["tab:blue", "lightgrey", "tab:orange"],
                                            N=256)

#from mcrnnoc.stats import figure_style

import sys

input_dir = "output/" + sys.argv[1]
n = input_dir.split("_n=")[-1].split(".")[0].split("_")[0]
n = int(n)
colorbar = 1

if len(sys.argv) > 2:
  colorbar = int(sys.argv[2])
end

input_filename = input_dir
u_vec = np.loadtxt(input_filename + ".txt")

mesh = UnitSquareMesh(n, n)
U = FunctionSpace(mesh, "DG", 0)
u = Function(U)
u.vector()[:] = u_vec
p = plot(u, wireframe=False, cmap=cmap_blue_orange)
if colorbar == 1:
  # https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
  plt.colorbar(p, fraction=0.046, pad=0.04)
end

plt.tight_layout()
plt.savefig(input_dir + ".pdf")
