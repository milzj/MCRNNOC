# Adapted from: https://github.com/milzj/FW4PDE/blob/main/examples/nonconvex/bilinear/bilinear_lusolver.py
import numpy as np

from fenics import *
from dolfin_adjoint import *
import moola

set_log_level(30)

from fw4pde.algorithms import FrankWolfe, MoolaBoxLMO
from fw4pde.problem import ScaledL1Norm, BoxConstraints
from fw4pde.stepsize import DemyanovRubinovOptimalStepSize

from mcrnnoc.examples import ProblemData
from mcrnnoc.examples import SolverOptions

import matplotlib.pyplot as plt

from mcrnnoc.sampler import TruncatedGaussianSampler
from mcrnnoc.random_field.exp_random_field import ExpRandomField

n = int(sys.argv[1])
now = str(sys.argv[2])
N = int(sys.argv[3])

import os
outdir = "output/"
outdir = outdir+"Risk_Neutral_Simulation_n="+str(n)+"_N_="+str(N)+"_date={}".format(now)
if not os.path.exists(outdir):
	os.makedirs(outdir)

problem_data = ProblemData()
solver_options = SolverOptions()

# solver options
options = solver_options.options
stepsize = DemyanovRubinovOptimalStepSize()

# problem data
lb = problem_data.lb
ub = problem_data.ub
beta = problem_data.beta
f = problem_data.f
g = problem_data.g
yd = problem_data.yd

# PDE
mesh = UnitSquareMesh(n,n)

U = FunctionSpace(mesh, "DG", 0)
V = FunctionSpace(mesh, "CG", 1)

u = Function(U)
y = TrialFunction(V)
v = TestFunction(V)

bc = DirichletBC(V, 0.0, "on_boundary")


sampler = TruncatedGaussianSampler()
exp_kappa = ExpRandomField(U)
num_rvs = exp_kappa.num_rvs


# objective functions
scaled_L1_norm = ScaledL1Norm(U,beta)

J = 0.0
for i in range(N):
    sample = sampler.sample(num_rvs)
    kappa_sample = exp_kappa.sample(sample)
    a = (kappa_sample*inner(grad(y), grad(v)) + g*y*u*v) * dx
    L = f*v*dx
    A, b  = assemble_system(a, L, bc)
    solver = LUSolver(A, "petsc")
    Y = Function(V)
    solver.solve(Y.vector(), b)
    j = assemble(0.5*inner(Y-yd,Y-yd)*dx)
    J += 1.0/(i+1.0)*(j-J)

control = Control(u)
rf = ReducedFunctional(J, control)

problem = MoolaOptimizationProblem(rf)
u_moola = moola.DolfinPrimalVector(u)

with stop_annotating():

    # constraints
    box_constraints = BoxConstraints(U, lb, ub)
    moola_box_lmo = MoolaBoxLMO(box_constraints.lb, box_constraints.ub, beta)

    solver = FrankWolfe(problem,
                    initial_point=u_moola,
                    nonsmooth_functional=scaled_L1_norm,
                    stepsize=stepsize,
                    lmo=moola_box_lmo,
                    options=options)

    sol = solver.solve()

    solution_final = sol["control_final"].data
    filename = outdir + "/" + "final_nominal_n={}_N={}".format(n,N)
    np.savetxt(filename + ".txt", solution_final.vector().get_local())

    p = plot(solution_final)
    plt.colorbar(p)
    plt.savefig(filename +  ".pdf", bbox_inches="tight")
    plt.savefig(filename +  ".png", bbox_inches="tight")
    plt.close()

    solution_best = sol["control_best"].data
    filename = outdir + "/" + "best_nominal_n={}_N={}".format(n,N)
    np.savetxt(filename + ".txt", solution_best.vector().get_local())

    p = plot(solution_best)
    plt.colorbar(p)
    plt.savefig(filename +  ".pdf", bbox_inches="tight")
    plt.savefig(filename +  ".png", bbox_inches="tight")
    plt.close()

    obj = problem.obj
    solution_final = sol["control_final"]
    obj(solution_final)
    gradient = obj.derivative(solution_final).primal()
    gradient_vec = gradient.data.vector()[:]
    filename = outdir + "/" + "final_gradient_vec_n={}".format(n)
    np.savetxt(filename+ ".txt", gradient_vec)


