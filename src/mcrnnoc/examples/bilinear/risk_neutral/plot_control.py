from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

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

p = plot(u)
plt.colorbar(p)
plt.savefig(input_dir + ".pdf", bbox_inches="tight")
