# Adapted from: https://github.com/milzj/FW4PDE/blob/main/examples/nonconvex/bilinear/bilinear_lusolver.py
import numpy as np

from fenics import *
from dolfin_adjoint import *
import moola

set_log_level(30)

from fw4pde.algorithms import FrankWolfe, MoolaBoxLMO
from fw4pde.problem import ScaledL1Norm, BoxConstraints
from fw4pde.stepsize import DunnScalingStepSize

from mcrnnoc.examples.linear import RandomLinearProblem
from mcrnnoc.examples.linear import LinearProblemData

from mcrnnoc.examples import SolverOptions
from mcrnnoc.criticality_measures import FEniCSCriticalityMeasures

import matplotlib.pyplot as plt
from mcrnnoc.examples.linear import plot_control
from mcrnnoc.stats import save_dict
from mcrnnoc.stats.figure_style import *

n = int(sys.argv[1])
now = str(sys.argv[2])

control_problem = RandomLinearProblem(n)

import os
outdir = "output/"
outdir = outdir+str(control_problem)+"_Nominal_Simulation_n="+str(n)+"_date={}".format(now)
if not os.path.exists(outdir):
	os.makedirs(outdir)

solver_options = SolverOptions()

# solver options
options = solver_options.options
stepsize = solver_options.stepsize

# PDE and problem data
mesh = UnitSquareMesh(n,n)
problem_data = LinearProblemData(mesh.mpi_comm())

lb = problem_data.lb
ub = problem_data.ub
beta = problem_data.beta
yd = problem_data.yd


U = FunctionSpace(mesh, "DG", 0)
V = FunctionSpace(mesh, "CG", 1)

u = Function(U)
y = TrialFunction(V)
v = TestFunction(V)

a = (inner(grad(y), grad(v))) * dx
L = u*v*dx
bc = DirichletBC(V, 0.0, "on_boundary")

A, b  = assemble_system(a, L, bc)

Y = Function(V)
solver = LUSolver(A, "petsc")
solver.solve(Y.vector(), b)

# objective functions
scaled_L1_norm = ScaledL1Norm(U,beta)
J = assemble(0.5*inner(Y-yd,Y-yd)*dx)

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
    filename = outdir + "/" + str(control_problem) + "_" + "final_nominal_n={}_{}".format(n, now)
    np.savetxt(filename + ".txt", solution_final.vector().get_local())

    plot_control(filename)

    # save relative path + filename of control
    relative_path = filename.split("/")
    relative_path = relative_path[1] + "/"  + relative_path[2]
    np.savetxt(filename + "_filename.txt", np.array([relative_path]), fmt = "%s")

    gradient_final = sol["gradient_final"].data

    plot_control(filename)

    # Comparing canonical criticality measure and dual gap
    cm = FEniCSCriticalityMeasures(U, lb, ub, beta)
    dual_gap = sol["dual_gap"]
    assert cm.canonical_map(solution_final, gradient_final) <= sqrt(dual_gap)

    solution_best = sol["control_best"].data
    filename = outdir + "/" + str(control_problem) + "_" + "best_nominal_n={}_{}".format(n, now)
    np.savetxt(filename + ".txt", solution_best.vector().get_local())

    plot_control(filename)

    gradient_final_vec = sol["gradient_final"].data.vector()[:]
    filename = outdir + "/" + str(control_problem) + "_"  + "gradient_final_vec_n={}_".format(n, now)
    np.savetxt(filename+ ".txt", gradient_final_vec)

    plot_control(filename)

    # save exit data
    filename = now + "_reference_solution_n={}_exit_data".format(n)
    # Transform moola objects to arrays
    sol["control_final"] = sol["control_final"].data.vector()[:]
    sol["control_best"] = sol["control_best"].data.vector()[:]
    sol["gradient_final"] = sol["gradient_final"].data.vector()[:]
    sol["gradient_best"] = sol["gradient_best"].data.vector()[:]

    save_dict(outdir, filename, sol)

