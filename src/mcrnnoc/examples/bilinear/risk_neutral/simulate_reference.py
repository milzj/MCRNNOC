from dolfin import *
from dolfin_adjoint import *
import moola
import numpy as np


from mcrnnoc.examples.bilinear import RandomBilinearProblem
from mcrnnoc.examples.solver_options import SolverOptions
from mcrnnoc.sampler import ReferenceTruncatedGaussianSampler

from mcrnnoc.random_problem import GlobalReducedSAAFunctional
from mcrnnoc.random_problem import RieszMap

from fw4pde.algorithms import FrankWolfe, MoolaBoxLMO
from fw4pde.problem import ScaledL1Norm, BoxConstraints

from mcrnnoc.stats import save_dict
from mcrnnoc.misc import criticality_measure


def simulate_reference(n, N, initial_control=None):

    set_working_tape(Tape())

    random_problem = RandomBilinearProblem(n)
    num_rvs = random_problem.num_rvs
    sampler = ReferenceTruncatedGaussianSampler(Nref=N, num_rvs=num_rvs, scramble=False)
    solver_options = SolverOptions()

    u = Function(random_problem.control_space)

    if initial_control != None:
        u = project(initial_control, random_problem.control_space)

    rf = GlobalReducedSAAFunctional(random_problem, u, sampler, N)

    beta = random_problem.beta
    lb = random_problem.lb
    ub = random_problem.ub

    riesz_map = RieszMap(random_problem.control_space)
    u_moola = moola.DolfinPrimalVector(u, riesz_map = riesz_map)

    problem = MoolaOptimizationProblem(rf, memoize=1)

    scaled_L1_norm = ScaledL1Norm(random_problem.control_space,beta)
    box_constraints = BoxConstraints(random_problem.control_space, lb, ub)
    moola_box_lmo = MoolaBoxLMO(box_constraints.lb, box_constraints.ub, beta)

    solver = FrankWolfe(problem,
                    initial_point=u_moola,
                    nonsmooth_functional=scaled_L1_norm,
                    stepsize=solver_options.stepsize,
                    lmo=moola_box_lmo,
                    options=solver_options.options)

    sol = solver.solve()

    return sol

if __name__ == "__main__":

    mpi_rank = MPI.comm_world.Get_rank()

    if mpi_rank == 0:

        import os, sys

        n = int(sys.argv[1])
        N = int(sys.argv[2])
        now = str(sys.argv[3])
        u_opt = None

        outdir = "output/"
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        outdir = outdir+"Reference_Simulation_n="+str(n)+"_N="+str(N)+"_date={}".format(now)
        os.makedirs(outdir)


    else:

        now = None
        outdir = None
        n = None
        N = None
        beta_filename = None
        u_opt = None


    now = MPI.comm_world.bcast(now, root=0)
    outdir = MPI.comm_world.bcast(outdir, root=0)

    n = MPI.comm_world.bcast(n, root=0)
    N = MPI.comm_world.bcast(N, root=0)

    u_opt = MPI.comm_world.bcast(u_opt, root=0)

    print("Homotopy method\n")
    ns_ = [n]+[2**i for i in range(5, int(np.log2(n)+1))]
    for n_ in sorted(list(set(ns_))):
        print("Homotopy method with n = {}".format(n_))
        sol = simulate_reference(n_, N, initial_control=u_opt)
        u_opt = sol["control_final"].data
        MPI.comm_world.barrier()


    if mpi_rank == 0:
        # save control
        filename = outdir + "/" + now + "_reference_solution_mpi_rank={}_N={}_n={}".format(mpi_rank, N,n)
        np.savetxt(filename + ".txt", u_opt.vector().get_local())

        # save relative path + filename of control
        relative_path = filename.split("/")
        relative_path = relative_path[1] + "/"+ relative_path[2]
        np.savetxt(filename + "_filename.txt", np.array([relative_path]), fmt = "%s")

        # save control as pvd
        file = File(MPI.comm_self, filename + ".pvd")
        file << u_opt

        # save exit data
        gradient_final_vec = sol["gradient_final"].data.vector()[:]
        filename = outdir + "/" + "gradient_final_vec_n={}".format(n)
        np.savetxt(filename+ ".txt", gradient_final_vec)

        # Transform moola objects to arrays
        filename = now + "_reference_solution_mpi_rank={}_N={}_n={}_exit_data".format(mpi_rank, N,n)
        sol["control_final"] = sol["control_final"].data.vector()[:]
        sol["control_best"] = sol["control_best"].data.vector()[:]
        sol["gradient_final"] = sol["gradient_final"].data.vector()[:]
        sol["gradient_best"] = sol["gradient_best"].data.vector()[:]

        save_dict(outdir, filename, sol)

