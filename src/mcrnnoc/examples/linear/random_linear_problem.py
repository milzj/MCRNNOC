from dolfin import *
from dolfin_adjoint import *
import numpy as np
from mcrnnoc.random_problem import RandomProblem
from .linear_problem_data import LinearProblemData
from mcrnnoc.random_field.exp_random_field import ExpRandomField
from mcrnnoc.random_field.options_random_field import OptionsRandomField

class RandomLinearProblem(RandomProblem):

    def __init__(self, n):

        set_working_tape(Tape())

        self.n = n

        # function spaces and functions
        mesh = UnitSquareMesh(MPI.comm_self, n, n)
        mpi_comm = mesh.mpi_comm()
        self.mpi_comm = mpi_comm
        V = FunctionSpace(mesh, "CG", 1)
        U = FunctionSpace(mesh, "DG", 0)

        self.V = V
        self.U = U

        self.y = Function(V)
        self.ytrial = TrialFunction(V)
        self.v = TestFunction(V)
        self.u = Function(U)

        # problem data
        problem_data = LinearProblemData(mpi_comm)

        self.lb = problem_data.lb
        self.ub = problem_data.ub
        self.beta = problem_data.beta
        self.yd = problem_data.yd
        self.f = problem_data.f
        self.bcs = DirichletBC(self.V, 0.0, "on_boundary")

        # random field
        options_random_field = OptionsRandomField().options
        self.len_scale = options_random_field["len_scale"]
        self.num_addends = options_random_field["num_addends"]
        self.num_rvs = options_random_field["num_addends"]**2
        self.kappa = ExpRandomField(U, options_random_field)

    def __str__(self):
        return "RandomLinearProblem"

    @property
    def control_space(self):
        return self.U

    def state(self, y, v, u, sample):
        """Linear PDE"""

        U = self.U
        bcs = self.bcs
        f = self.f
        _kappa = self.kappa.sample_vec(sample)

        kappa = Function(U)
        kappa.vector().set_local(_kappa)

        F = (kappa*inner(grad(y), grad(v)) - u*v - f*v) * dx
        solve(F == 0, y, bcs=self.bcs)


    def __call__(self, u, sample):

        y = self.y
        yd = self.yd
        v = self.v
        self.state(y, v, u, sample)

        return assemble(0.5*inner(y-yd,y-yd)*dx)




