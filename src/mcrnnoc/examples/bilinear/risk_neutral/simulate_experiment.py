# Adapted from https://github.com/milzj/SAA4PDE/blob/semilinear_complexity/simulations/semilinear_complexity/simulate_experiment.py

from dolfin import *
from dolfin_adjoint import *
import moola
import numpy as np

from scipy.stats import qmc


from mcrnnoc.examples.bilinear import RandomBilinearProblem
from mcrnnoc.examples.solver_options import SolverOptions
from mcrnnoc.sampler import ReferenceTruncatedGaussianSampler
from mcrnnoc.sampler import TruncatedGaussianSampler
from mcrnnoc.sampler import TruncatedGaussianSobolSampler

from mcrnnoc.random_problem import LocalReducedSAAFunctional
from mcrnnoc.random_problem import RieszMap
from mcrnnoc.prox import prox_box_l1

from mcrnnoc.misc.criticality_measure import criticality_measure


from fw4pde.algorithms import FrankWolfe, MoolaBoxLMO
from fw4pde.problem import ScaledL1Norm, BoxConstraints

from mcrnnoc.stats import save_dict
from mcrnnoc.misc import criticality_measure

import warnings

class SAAProblems(object):


    def __init__(self, date=-1, experiment=None, Nref=-1, num_reps=30, experiment_name=None):

        self.date = date
        self.experiment = experiment
        self.num_reps = num_reps
        self.Nref = Nref
        self.experiment_name = experiment_name

        self.mpi_size = MPI.comm_world.Get_size()
        self.mpi_rank = MPI.comm_world.Get_rank()
        self.LocalStats = {}

        self.seeds()
        self.divide_simulations()

        random_problem = RandomBilinearProblem(8)
        num_rvs = random_problem.num_rvs
        self.reference_sampler = ReferenceTruncatedGaussianSampler(Nref=Nref, num_rvs=num_rvs)

    def seeds(self):

        Seeds = {}
        num_reps = self.num_reps

        seed = self.Nref

        for r in range(1, 1+num_reps):

            Seeds[r] = {}

            for e in self.experiment[("n_vec", "N_vec")]:
                n, N = e

                seed += 1*N+1
                Seeds[r][e] = seed

        if np.__version__ == '1.12.1':
            period = 2**32-1
        else:
            period = 2**32-1

        assert seed <= period, "Period of random number generator (might) too small."

        self.Seeds = Seeds

    def divide_simulations(self):

        mpi_size = self.mpi_size
        mpi_rank = self.mpi_rank
        num_reps = self.num_reps

        reps = np.array(range(1,1+num_reps))

        Reps = np.array_split(reps, mpi_size)
        self.Reps = Reps


    def local_solve(self, sampler, n, number_samples, initial_control=None):

        set_working_tape(Tape())
        solver_options = SolverOptions()

        random_problem = RandomBilinearProblem(n)
        sampler.num_rvs = random_problem.num_rvs

        u = Function(random_problem.control_space)

        if initial_control != None:
            u = project(initial_control, random_problem.control_space)


        rf = LocalReducedSAAFunctional(random_problem, u, sampler, number_samples, mpi_comm = MPI.comm_self)

        assert rf.mpi_size == 1

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

        return sol, sol["dual_gap"], sol["control_final"].data, sol["gradient_final"].data

    def criticality_measure(self, control_vec, gradient_vec, n, Nref):
        """Evaluate reference gap function and criticality measure without parallelization."""

        set_working_tape(Tape())
        solver_options = SolverOptions()

        random_problem = RandomBilinearProblem(n)

        sampler = self.reference_sampler

        u = Function(random_problem.control_space)
        u.vector()[:] = control_vec

        rf = LocalReducedSAAFunctional(random_problem, u, sampler, Nref, mpi_comm = MPI.comm_self)
        print("Nref", Nref)

        beta = random_problem.beta
        lb = random_problem.lb
        ub = random_problem.ub

        riesz_map = RieszMap(random_problem.control_space)
        v_moola = moola.DolfinPrimalVector(u, riesz_map = riesz_map)
        #v_moola = moola.DolfinPrimalVector(u)

        problem = MoolaOptimizationProblem(rf, memoize=0)
        obj = problem.obj
        obj(v_moola)

        deriv = obj.derivative(v_moola)
        grad = deriv.primal()
        grad_vec = grad.data.vector().get_local()

        criticality_measures = []

        # reference gap function
        scaled_L1_norm = ScaledL1Norm(random_problem.control_space,beta)
        box_constraints = BoxConstraints(random_problem.control_space, lb, ub)
        moola_box_lmo = MoolaBoxLMO(box_constraints.lb, box_constraints.ub, beta)


        v = v_moola.copy().zero()
        u_minus_v = v_moola.copy().zero()
        moola_box_lmo.solve(grad, v)

        u_minus_v.assign(v_moola)
        u_minus_v.axpy(-1.0, v)
        dual_gap = deriv.apply(u_minus_v) + \
                    scaled_L1_norm(u) - \
                    scaled_L1_norm(v.data)

        criticality_measures.append(dual_gap)

        # reference crit measure
        g_vec = prox_box_l1(control_vec-grad_vec, box_constraints.lb, box_constraints.ub, beta)
        prox_grad = Function(random_problem.control_space)
        prox_grad.vector()[:] = g_vec
        criticality_measures.append(errornorm(u, prox_grad, degree_rise = 0))

        # crit measure
        gg_vec = prox_box_l1(control_vec-gradient_vec, box_constraints.lb, box_constraints.ub, beta)
        prox_gradient = Function(random_problem.control_space)
        prox_gradient.vector()[:] = gg_vec
        criticality_measures.append(errornorm(u, prox_gradient, degree_rise = 0))

        return criticality_measures


    def simulate_mpi(self):

        LocalStats = {}
        LocalSols = {}
        mpi_rank = self.mpi_rank

        sampler = TruncatedGaussianSobolSampler()
        sampler = TruncatedGaussianSampler()

        Nref = int(self.Nref)

        for r in self.Reps[mpi_rank]:
            E = {}
            S = {}

            for e in self.experiment[("n_vec", "N_vec")]:
                n, N = e
                print("r, n, N", r, n, N)

                seed = self.Seeds[r][e]
                sampler._seed = seed

                assert sampler.seed == seed
                if self.experiment_name.find("Synthetic") != -1:
                    warnings.warn("Simulation output is synthetic." +
                        " This is a verbose mode used to generate test data for plotting purposes.")
                    np.random.seed(seed)

                    qmc_sampler = qmc.Sobol(d=1, scramble=True, seed=seed)
                    m = int(np.log2(N))
                    errors = qmc_sampler.random_base2(m=m)-0.5
                    errors = abs(np.mean(errors))
                    sol = np.random.randn(3)

                else:

                    u_opt = None
                    for n_ in [32, n]:
                        print("Homotopy method with n = {}".format(n_))
                        print("r, n, N", r, n_, n, N)
                        sol, dual_gap, u_opt, grad_opt = self.local_solve(sampler, n_, N, initial_control=u_opt)
                        sampler._seed = seed
                        cm_value = criticality_measure(u_opt, grad_opt, -1.0 ,1.0, 1e-3)
                        print("sqrt(dual_gap)={}".format(sqrt(dual_gap)))
                        print("cm_value={}".format(cm_value))


                    print("sqrt(dual_gap)={}".format(sqrt(dual_gap)))
                    print("cm_value={}".format(cm_value))
    
                    errors = self.criticality_measure(u_opt.vector()[:], grad_opt.vector()[:], n, Nref)
                    errors.append(dual_gap)
                    sol = u_opt.vector()[:]
                    print("errors", errors)


                E[e] = errors
                S[e] = sol

            LocalStats[r] = E
            LocalSols[r] = S

        self.LocalStats = LocalStats
        self.LocalSols = LocalSols



    def save_mpi(self, now, outdir):
        filename = now + "_mpi_rank=" + str(MPI.comm_world.Get_rank())
        save_dict(outdir, filename, self.LocalStats)
        filename = now + "_solutions_mpi_rank=" + str(MPI.comm_world.Get_rank())
        save_dict(outdir, filename, self.LocalSols)



if __name__ == "__main__":

    import sys, os

    from experiments import Experiments
    from mcrnnoc.stats import save_dict

    if MPI.comm_world.Get_rank() == 0:

        # sys.argv
        date = sys.argv[1]
        experiment_name = sys.argv[2]
        Nref = int(sys.argv[3])

        # output dir
        outdir = "output/Experiments/" + experiment_name + "_" + date

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        experiment = Experiments()(experiment_name)

        # save experiment
        filename = experiment_name
        save_dict(outdir, filename, {experiment_name: experiment})

        ## save relative path + filename of control
        np.savetxt(outdir  + "/" + filename  + "_filename.txt", np.array([outdir]), fmt = "%s")


    else:
        date = None
        experiment_name = None
        outdir = None
        experiment = None
        Nref = None


    # bcast
    date = MPI.comm_world.bcast(date, root=0)
    experiment_name = MPI.comm_world.bcast(experiment_name, root=0)
    outdir = MPI.comm_world.bcast(outdir, root=0)
    experiment = MPI.comm_world.bcast(experiment, root=0)
    Nref = MPI.comm_world.bcast(Nref, root=0)


    saa_problems = SAAProblems(date=date,experiment=experiment, experiment_name=experiment_name,Nref = Nref)

    saa_problems.simulate_mpi()
    saa_problems.save_mpi(date, outdir)
