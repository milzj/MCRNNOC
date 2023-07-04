from dolfin import *
from dolfin_adjoint import *

from mcrnnoc.random_problem import RandomProblem
from mcrnnoc.examples import ProblemData
from mcrnnoc.random_field.exp_random_field import ExpRandomField
from mcrnnoc.random_field.options_random_field import OptionsRandomField

class RandomBilinearProblem(RandomProblem):

    def __init__(self, n):

        set_working_tape(Tape())

        self.n = n

        # function spaces and functions
        mesh = UnitSquareMesh(MPI.comm_self, n, n)
        mpi_comm = mesh.mpi_comm()
        V = FunctionSpace(mesh, "CG", 1)
        U = FunctionSpace(mesh, "DG", 0)

        self.V = V
        self.U = U

        self.y = Function(V)
        self.v = TestFunction(V)
        self.u = Function(U)

        # problem data
        problem_data = ProblemData(mpi_comm)
        self.mpi_comm = mpi_comm

        self.lb = problem_data.lb
        self.ub = problem_data.ub
        self.beta = problem_data.beta
        self.f = problem_data.f
        self.g = problem_data.g
        self.yd = problem_data.yd
        self.bcs = DirichletBC(self.V, 0.0, "on_boundary")

        # random field
        options_random_field = OptionsRandomField().options
        self.len_scale = options_random_field["len_scale"]
        self.num_addends = options_random_field["num_addends"]
        self.num_rvs = options_random_field["num_addends"]**2
        self.kappa = ExpRandomField(U, options_random_field)

    @property
    def control_space(self):
        return self.U

    def state(self, y, v, u, sample):
        """Bilinear PDE"""

        g = self.g
        f = self.f
        bcs = self.bcs
        kappa = self.kappa.sample(sample)

        F = (kappa*inner(grad(y), grad(v)) + g*y*u*v - f*v) * dx
        solve(F == 0, y, bcs=self.bcs)


    def __call__(self, u, sample):

        y = self.y
        yd = self.yd
        v = self.v
        self.state(y, v, u, sample)

        return assemble(0.5*inner(y-yd,y-yd)*dx)




