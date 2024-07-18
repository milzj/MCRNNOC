# Adapted from https://github.com/milzj/SAA4PDE/blob/semilinear_complexity/simulations/semilinear_complexity/simulate_experiment.py

from dolfin import *
from dolfin_adjoint import *
set_log_level(10)

import moola
import numpy as np

from scipy.stats import qmc

from mcrnnoc.examples.linear import RandomLinearProblem
from mcrnnoc.examples.solver_options import SolverOptions
from mcrnnoc.sampler import ReferenceTruncatedGaussianSampler
from mcrnnoc.sampler import TruncatedGaussianSampler
from mcrnnoc.sampler import TruncatedGaussianSobolSampler

from mcrnnoc.random_problem import LocalReducedSAAFunctional
from mcrnnoc.random_problem import RieszMap
from mcrnnoc.criticality_measures import FEniCSCriticalityMeasures

from fw4pde.algorithms import FrankWolfe, MoolaBoxLMO
from fw4pde.problem import ScaledL1Norm, BoxConstraints

from mcrnnoc.stats import save_dict

import warnings

class SAAProblems(object):


    def __init__(self, date=-1, experiment=None, Nref=-1, num_reps=40, experiment_name=None, scramble=True):

        self.date = date
        self.experiment = experiment
        self.num_reps = num_reps
        self.Nref = Nref
        self.experiment_name = experiment_name
        self.scramble = scramble

        self.mpi_size = MPI.comm_world.Get_size()
        self.mpi_rank = MPI.comm_world.Get_rank()
        self.LocalStats = {}

        self.divide_simulations()

    def divide_simulations(self):

        mpi_size = self.mpi_size
        mpi_rank = self.mpi_rank
        num_reps = self.num_reps

        reps = np.array(range(1,1+num_reps))

        Reps = np.array_split(reps, mpi_size)
        self.Reps = Reps


    def saa_gradient(self, sampler, n, number_samples, initial_control=None):

        set_working_tape(Tape())
        solver_options = SolverOptions()

        random_problem = RandomLinearProblem(n)
        sampler.num_rvs = random_problem.num_rvs

        u = Function(random_problem.control_space)

        if initial_control != None:
            u = project(initial_control, random_problem.control_space)

        rf = LocalReducedSAAFunctional(random_problem, u, sampler, number_samples, mpi_comm = MPI.comm_self)

        assert rf.mpi_size == 1

        riesz_map = RieszMap(random_problem.control_space)
        u_moola = moola.DolfinPrimalVector(u, riesz_map = riesz_map)

        riesz_map = RieszMap(random_problem.control_space)
        u_moola = moola.DolfinPrimalVector(u, riesz_map = riesz_map)
        problem = MoolaOptimizationProblem(rf, memoize=1)

        obj = problem.obj
        obj(u_moola)
        derivative = obj.derivative(u_moola)
        gradient = derivative.primal()

        return gradient.data

    def local_solve(self, sampler, n, number_samples, initial_control=None):

        set_working_tape(Tape())
        solver_options = SolverOptions()

        random_problem = RandomLinearProblem(n)
        sampler.num_rvs = random_problem.num_rvs

        u = Function(random_problem.control_space)
        u.vector()[:] = 1

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


    def reference_criticality_measure(self, control_vec, n, Nref, gradient_vec=None, sampler=None):
        """Evaluate reference gap function and criticality measure without parallelization."""

        set_working_tape(Tape())
        solver_options = SolverOptions()

        random_problem = RandomLinearProblem(n)

        u = Function(random_problem.control_space)
        u.vector().set_local(control_vec)
        num_rvs = random_problem.num_rvs

        if sampler == None:
            sampler =  ReferenceTruncatedGaussianSampler(Nref=Nref, num_rvs=num_rvs, scramble=self.scramble)
        else:
            sampler.num_rvs = random_problem.num_rvs

        rf = LocalReducedSAAFunctional(random_problem, u, sampler, Nref, mpi_comm = MPI.comm_self)

        beta = random_problem.beta
        lb = random_problem.lb
        ub = random_problem.ub

        riesz_map = RieszMap(random_problem.control_space)
        v_moola = moola.DolfinPrimalVector(u, riesz_map = riesz_map)

        problem = MoolaOptimizationProblem(rf, memoize=0)
        obj = problem.obj
        obj(v_moola)

        deriv = obj.derivative(v_moola)
        grad = deriv.primal()
        grad_vec = grad.data.vector().get_local()

        criticality_measures = []

        cm = FEniCSCriticalityMeasures(random_problem.control_space, lb, ub, beta)
        criticality_measures += [cm.gap(u, grad, deriv)]
        criticality_measures += [cm.rgap(u, grad, deriv)]
        criticality_measures += [cm.canonical_map(u, grad.data)]

        return criticality_measures

    def saa_gradient_error(self, control_vec, n, Nref, gradient_vec=None, sampler=None, reference_gradient_vec=np.array([None])):
        """Evaluate reference gap function and criticality measure without parallelization."""

        set_working_tape(Tape())
        random_problem = RandomLinearProblem(n)

        if reference_gradient_vec.any() == None:    
            solver_options = SolverOptions()
        
            u = Function(random_problem.control_space)
            u.vector().set_local(control_vec)
            num_rvs = random_problem.num_rvs
    
            if sampler == None:
                sampler = ReferenceTruncatedGaussianSampler(Nref=Nref, num_rvs=num_rvs, scramble=self.scramble)
            else:
                sampler.num_rvs = random_problem.num_rvs
    
            rf = LocalReducedSAAFunctional(random_problem, u, sampler, Nref, mpi_comm = MPI.comm_self)
            print("Nref", Nref)
    
            riesz_map = RieszMap(random_problem.control_space)
            u_moola = moola.DolfinPrimalVector(u, riesz_map = riesz_map)
    
            problem = MoolaOptimizationProblem(rf, memoize=0)
            obj = problem.obj
            obj(u_moola)

            deriv = obj.derivative(u_moola)
            grad = deriv.primal().data
            reference_gradient_vec = grad.vector()[:]

        else:
            grad = Function(random_problem.control_space)
            grad.vector().set_local(reference_gradient_vec)

        saa_grad = Function(random_problem.control_space)
        saa_grad.vector().set_local(gradient_vec)

        return errornorm(grad, saa_grad, degree_rise = 0), reference_gradient_vec


    def simulate_mpi(self):

        LocalStats = {}
        LocalSols = {}
        mpi_rank = self.mpi_rank
        print("-------------------------")
        print("MPI RANK {}".format(mpi_rank))
        print("-------------------------")


        Nref = int(self.Nref)

        indicator_fixed_control = -1

        for r in self.Reps[mpi_rank]:
            E = {}
            S = {}
            print(self.experiment_name)
            reference_gradient_vec = np.array([None])

            for e in self.experiment[("n_vec", "N_vec")]:
                n, N = e
                print("r, n, N", r, n, N)


                if self.experiment_name.find("Synthetic") != -1:
                    warnings.warn("Simulation output is synthetic." +
                        " This is a verbose mode used to generate test data for plotting purposes.")

                    qmc_sampler = qmc.Sobol(d=1, scramble=True, seed=1234)
                    m = int(np.log2(N))
                    errors = qmc_sampler.random_base2(m=m)-0.5
                    errors = abs(np.mean(errors))
                    sol = np.random.randn(3)

                else:

                    set_working_tape(Tape())
                    random_problem = RandomLinearProblem(n)
                    num_rvs = random_problem.num_rvs
                    if self.experiment_name.find("Fixed_Control") != -1:



                        U = random_problem.control_space
                        u_opt = Expression('x[0] < 0.5 ? -1.0 : 1.0', degree=0, mpi_comm=MPI.comm_self)
                        saa_gradient = self.saa_gradient(sampler, n, N, initial_control=u_opt)

                        u_opt = project(u_opt, U)
                        u_opt = u_opt.vector()[:]

                        errors, reference_gradient_vec = self.saa_gradient_error(u_opt, n, Nref,\
                                            gradient_vec = saa_gradient.vector()[:], reference_gradient_vec = reference_gradient_vec)

                        sol = u_opt

                    else:

                        u_opt = None
                        ns_ = [n]+[2**i for i in range(5, int(np.log2(n)+1))]
                        for n_ in sorted(list(set(ns_))):
                            print("Homotopy method with n = {}".format(n_))
                            print("r, n_, n, N", r, n_, n, N)
                            sampler = TruncatedGaussianSampler(r-1, N, num_rvs, self.num_reps)
                            sol, dual_gap, u_opt, grad_opt = self.local_solve(sampler, n_, N, initial_control=u_opt)

                        u_opt = u_opt.vector()[:]

                        errors = self.reference_criticality_measure(u_opt, n, Nref, gradient_vec = grad_opt.vector()[:])
                        errors.append(dual_gap)
                        sol = u_opt

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
