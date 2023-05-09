from dolfin import *
from dolfin_adjoint import *

from random_problem import RandomProblem

from mcrnnoc.examples import ProblemData
from mcrnnoc.random_field.exp_random_field import ExpRandomField

class RandomSemilinearProblem(RandomProblem):

	def __init__(self, n):

		set_working_tape(Tape())

		self.n = n

        # function spaces and functions
		mesh = UnitSquareMesh(MPI.comm_self, n, n)
		V = FunctionSpace(mesh, "CG", 1)
		U = FunctionSpace(mesh, "DG", 0)

		self.V = V
		self.U = U

		self.y = Function(V)
        self.s = TrailFunction(V)
		self.v = TestFunction(V)
		self.u = Function(U)

        # problem data
        self.lb = problem_data.lb
        self.ub = problem_data.ub
        self.beta = problem_data.beta
        self.f = problem_data.f
        self.g = problem_data.g
        self.yd = problem_data.yd
		self.bcs = DirichletBC(self.V, 0.0, "on_boundary")

        # random field
        self.kappa = ExpRandomField(U)

	@property
	def control_space(self):
		return self.U


	def state(self, y, s, v, u, sample):
        """Bilinear PDE"""

        g = self.g
        f = self.f
        bcs = self.bcs
        kappa = self.kappa.sample(sample)

        a = (kappa*inner(grad(s), grad(v)) + g*s*u*v) * dx
        L = f*v*dx

        A, b  = assemble_system(a, L, bcs)
        solver = LUSolver(A, "petsc")
        solver.solve(y.vector(), b)

	def __call__(self, u, sample):

		y = self.y
		s = self.s
		yd = self.yd
		v = self.v
		self.state(y, s, v, u, sample)

        return assemble(0.5*inner(y-yd,y-yd)*dx)


