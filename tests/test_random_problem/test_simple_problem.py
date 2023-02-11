"""
mpiexec -n 4 python test_simple_problem.py
mpiexec -n 4 python -m pytest test_simple_problem.py

"""

import pytest

from dolfin import *
from dolfin_adjoint import *
import numpy as np

from problem import Regularizer

from random_problem import GlobalReducedSAAFunctional

from random_problem import RandomProblem
from random_problem import Sampler

class SimpleSampler(Sampler):

	def __init__(self, sample):
		self._sample = sample

	def sample(self, sample_index):
		return self._sample[sample_index]

class SimpleProblem(RandomProblem):

	def __init__(self):

		set_working_tape(Tape())

		n = 32

		mesh = UnitSquareMesh(MPI.comm_self, n, n)
		U = FunctionSpace(mesh, "DG", 0)

		self.U = U

	@property
	def control_space(self):
		return self.U

	def __call__(self, u, sample):
		return assemble(Constant(sample)*u**2*dx)

class ExactSimpleProblem(object):

	def __init__(self):

		set_working_tape(Tape())


	def exact_mean(self, sample, u):
		self.regularizer = Regularizer(u)
		return 2.0*np.mean(sample)*self.regularizer(u)

	def exact_derivative(self, sample, u):
		self.regularizer(u)
		return np.mean(sample)*2.0*self.regularizer.derivative()

def test_simple_problem():

	rtol = 1e-14

	random_problem, exact_problem = SimpleProblem(), ExactSimpleProblem()

	np.random.seed(1234)
	sample = np.random.randn(13)
	number_samples = len(sample)
	sampler = SimpleSampler(sample)

	u = Function(random_problem.control_space)
	u_expr = [Expression("sin(pi*x[0])*sin(pi*x[1])", degree = 3),
			Expression("cos(pi*x[0])*exp(pi*x[1])", degree = 3),
			Constant(1.0)]
	u.interpolate(u_expr[0])


	global_rf = GlobalReducedSAAFunctional(random_problem, u, sampler, number_samples)


	for i in range(len(u_expr)):

		computed_mean = global_rf(u)
		true_mean = exact_problem.exact_mean(sample,u)

		assert abs(computed_mean-true_mean)/true_mean <= rtol

		computed_derivative = global_rf.derivative().vector()

		true_derivative = exact_problem.exact_derivative(sample, u)

		assert np.linalg.norm(computed_derivative-true_derivative, ord=np.inf)/ \
				max(1.0,np.linalg.norm(true_derivative, ord=np.inf)) < rtol

		u.interpolate(u_expr[1])



if __name__ == "__main__":

	test_simple_problem()
