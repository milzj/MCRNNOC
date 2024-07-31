import dolfin

class Regularizer(object):
	"""Implements smooth control regularizer.

	Implements .5*u**2*dx and its first and second derivatives.

	Note:
		call function before derivative computation
		call derivative before hessian computation
	"""
	def __init__(self, control):

		self.u = control

	def __call__(self, u):

		self.u.assign(u)
		self.j = .5*self.u**2*dolfin.dx

		return dolfin.assemble(self.j)

	def derivative(self):
		"""

		Returns:
		--------
			

		"""

		self.d = dolfin.derivative(self.j, self.u)

		return dolfin.assemble(self.d)

	def hessian(self, direction):
		"""Computed Hessian times direction.

		The derivative is an element of U* = L(U,R). Its derivative
		(here called hessian) is an element of L(U,U*). The function returns
		the hessian evaluated in the direction `direction.`

		Returns:
		--------
			hessian : dolfin vector
				Hessian action
		"""

		H = dolfin.derivative(self.d, self.u, direction)

		return dolfin.assemble(H)
