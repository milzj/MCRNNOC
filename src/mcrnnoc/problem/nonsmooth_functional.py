
import moola
import numpy as np
from prox import diag_grad_prox_box_l1
from regularizer import Regularizer

class NonsmoothFunctional(object):
	"""

	References:
	-----------
		moola developers: https://github.com/funsim/moola/blob/master/moola/problem/functional.py

	"""


	def __init__(self, rf, v_moola, alpha, beta, lb=-np.inf, ub=np.inf):
		"""
		TODO: No super needed?
		"""
		self.rf = rf
		self.alpha = alpha
		self.beta = beta
		self.lb_vec = lb
		self.ub_vec = ub

		self.u_moola = v_moola.copy()
		self.T = v_moola.copy()
		self.__direction = v_moola.copy()
		self.regularizer = Regularizer(v_moola.data.copy())

	def prox(self, v_vec):
		"""Computes proximal operator and its derivative

		
		Note: Computing the proximal operator of dolfin vectors
		is rather time consuming. The function assumes that
		the input `v_vec` is a numpy vector.

		Parameters:
		-----------
			v_vec : numpy.ndarray
				input array
		Returns:
		--------
			
			

		References:
		-----------
		p. 818 in

		E. Casas, R. Herzog, and G. Wachsmuth, Optimality conditions
		and error analysis of semilinear elliptic control problems with
		L1 cost functional, SIAM J. Optim., 22 (2012),
		pp. 795–820, https://doi.org/10.1137/110834366.
		"""

		return diag_grad_prox_box_l1(v_vec, self.lb_vec, self.ub_vec, self.beta/self.alpha)

	def Prox(self, u_moola, v_moola):
		self.prox_vec, self.grad_prox = self.prox(v_moola.data.vector().get_local())

		u_moola.data.vector().set_local(self.prox_vec)
		u_moola.bump_version()


	def __call__(self, v_moola):
		"""
		
		Evaluates nonsmooth objective function: 
			rf(v) + alpha*0.5*norm(v,L2)**2 + beta*norm(v,L1)

		Note: We assume that the control space is DG0, that is, the space
		of piecewise constant nodal functions. This assumption is used to 
		evaluate the L1-norm. If M is the DG0 control mass matrix, then
		norm(v, L1) = norm(M*v, l1) (since the supports of distinct DG0 basis functions 
		are disjoint and their union is the discretized computational domain.)
		We compute M*v by computing the derivative of .5*v**2*dx.

		Returns:
		--------
			val : float
				composite objective function value, that is, 
				rf(v) + alpha*0.5*norm(v,L2)**2 + beta*norm(v,L1)
		"""
		# Call prox

		moola.events.increment("Functional evaluation")

		smooth_regularizer = self.regularizer(v_moola.data)

		deriv_regularizer = self.regularizer.derivative()
		deriv_regularizer.abs()
		nonsmooth_regularizer = deriv_regularizer.sum()

		return self.rf(v_moola.data) \
			+ self.alpha*smooth_regularizer \
			+ self.beta*nonsmooth_regularizer

	def derivative(self, v_moola):
		"""Computes derivative of smooth objective function.

		We call moola's convert_to_moola_dual_vector to convert
		the derivative to a moola dual vector.

		References:
		-----------
			pyadjoint developers: https://github.com/dolfin-adjoint/pyadjoint/blob/master/pyadjoint/optimization/moola_problem.py#L78
			moola developers: https://github.com/funsim/moola/blob/79ffc8d86dff383762063970358b8aebfe373896/moola/adaptors/adaptor.py
		"""

		moola.events.increment("Derivative evaluation")

		self.rf(v_moola.data)

		_deriv = self.rf.derivative()

		deriv = moola.convert_to_moola_dual_vector(_deriv, v_moola)

		return deriv

	def normal_map(self, u_moola, v_moola):
		"""Evaluates the dual element of the normal map.

		We implement eq. (3.3) in Pieper (2015).

		The primal of the normal map is alpha*v + grad F(prox(v)). The function
		computes the corresponding dual element. The dual element is equal to
		the inverse Riesz mapping applied ot the primal element.

		The proximal operator of v_moola is computed and stored in u_moola, that is,
		we compute prox(v_moola) and store the result in u_moola. Then we call 
		u_moola.bump_version().

		We call moola's convert_to_moola_dual_vector. 

		Parameters:
		-----------
			v_moola : moola.DolfinPrimalVector
				Iterate in semismooth Newton method
			u_moola : moola.DolfinPrimalVector
				Proximal operator of iterate (v_moola)

		Returns:
		--------
			deriv : moola.DolfinDualVector
				The dual element of the normal mapping.

		References:
		-----------

		K. Pieper, Finite element discretization and efficient numerical solution of
		elliptic and parabolic sparse control problems, Dissertation, TUM, Munich, 2015,
		http://mediatum.ub.tum.de/node?id=1241413
		"""

		self.prox_vec, self.grad_prox = self.prox(v_moola.data.vector().get_local())
		self.T.data.vector().set_local(self.grad_prox)

		u_moola.data.vector().set_local(self.prox_vec)
		u_moola.bump_version()

		self.rf(u_moola.data) # Should not be called here
		deriv1 = self.rf.derivative()

		self.regularizer(v_moola.data)
		deriv2 = self.regularizer.derivative()

		d = deriv1.vector() # Updates deriv1
		d.axpy(self.alpha, deriv2)

		deriv = moola.convert_to_moola_dual_vector(deriv1, v_moola)

		return deriv

	def derivative_normal_map(self, u_moola, v_moola):
		"""Computes the second derivative of the normal mapping.

		The function returns a function computing Hessian actions. The result
		of the Hessian action is an element of U* = L(U,IR).

		References:
		-----------
		pyadjoint developers: https://github.com/dolfin-adjoint/pyadjoint/blob/master/pyadjoint/optimization/moola_problem.py#L93

		p. 2092 in 

		F. Mannel and A. Rund, A hybrid semismooth quasi-Newton method for
		nonsmooth optimal control with PDEs, Optim. Eng., 22 (2021), pp. 2087–2125, 
		https://doi.org/10.1007/s11081-020-09523-w

		Prop. 3.11
			
		K. Pieper, Finite element discretization and efficient numerical solution of
		elliptic and parabolic sparse control problems, Dissertation, TUM, Munich, 2015,
		http://mediatum.ub.tum.de/node?id=1241413
		"""

		#self(v_moola)
		#self.rf(u_moola.data)

		def _hessian(direction):
			"""Computes Hessian actions.

			We have DG(u) = alpha*I + Hessian(f1)T. Here T is the diagonal of 
			the proximal operator's subgradient. The function computes the
			Hessian action:

				DG(u)[direction] = alpha*I[direction] + Hessian(f1) (T[direction]).
			"""
			moola.events.increment("Hessian evaluation")

			self.__direction.assign(direction)
			self.restrict(self.__direction)
			self.__direction.bump_version()

			self.rf(u_moola.data)
			hessian1 = self.rf.hessian(self.__direction.data)

			hessian2 = self.regularizer.hessian(direction.data)

			h = hessian1.vector()
			h.axpy(self.alpha, hessian2)

			return moola.convert_to_moola_dual_vector(hessian1, v_moola)

		return _hessian

	def restrict(self, v_moola):
		"""Evaluates directional derivative of proximal operator in the direction `v_moola.`

		Restricts input primal function v_moola to Hilbert space generated by T.

		scale operates componentwise.
		"""
		v_moola.scale(self.T.data.vector())



