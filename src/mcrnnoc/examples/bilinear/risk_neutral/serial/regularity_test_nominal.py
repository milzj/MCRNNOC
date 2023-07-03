import fenics
import numpy as np
import fw4pde
from mcrnnoc.examples import ProblemData
from mcrnnoc.stats import regularity_test
import sys, os

n = int(sys.argv[1])
now = str(sys.argv[2])

outdir = "output/"
outdir = outdir+"Nominal_Simulation_n="+str(n)+"_date={}".format(now)

problem_data = ProblemData()
beta = problem_data.beta

mesh = fenics.UnitSquareMesh(n,n)
U = fenics.FunctionSpace(mesh, "DG", 0)
gradient = fenics.Function(U)

gradient_vec = np.loadtxt(outdir + "/final_gradient_vec_n={}.txt".format(n))
gradient.vector()[:] = gradient_vec

regularity_test(gradient,
                beta,
                logspace_start=-11,
                logspace_stop=0,
                ndrop=3,
                figure_name=outdir + "/regularity_test_n={}".format(n))

