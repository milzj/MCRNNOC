from mcrnnoc.criticality_measures import CriticalityMeasures
import fenics
import fw4pde
from mcrnnoc.criticality_measures.prox import prox_box_l1

class FEniCSCriticalityMeasures(CriticalityMeasures):

    def __init__(self,function_space,lb,ub,beta,tau=1.0, solver_type="default"):

        _lb = fenics.project(lb, function_space, solver_type=solver_type)
        _ub = fenics.project(ub, function_space, solver_type=solver_type)

        lb_vec = _lb.vector().get_local()
        ub_vec = _ub.vector().get_local()

        self.function_space = function_space

        super().__init__(lb_vec, ub_vec, beta, tau=tau)

    def canonical_map(self, u, g):
        """Computes the L2-norm of the canonical residual."""

        u_vec = u.vector().get_local()
        g_vec = g.vector().get_local()

        self.canonical_residual(u_vec,g_vec)

        v = fenics.Function(self.function_space)
        v.vector().set_local(self._canonical_residual)

        return fenics.norm(v, norm_type = "L2")

    def normal_map(self, v, g):
        """Computes the L2-norm of the normal residual."""
        v_vec = v.vector().get_local()
        g_vec = g.vector().get_local()

        self.normal_residual(v_vec, g_vec)

        w = fenics.Function(self.function_space)
        w.vector().set_local(self._normal_residual)

        return fenics.norm(w, norm_type = "L2")

    def rgap(self, u, g, deriv):
        """Computes regularized gap function

        (g,u-w)_H + \psi(u)-\psi(w)-(nu/2)\|u-w\|_H^2,

        where w = prox_{\psi/tau}(x-(1/tau)*g) and \psi(u) = beta*\|u\|_1

        Parameters:
        -----------
            u : fenics.Function
                control
            g : moola.Function
                gradient
            deriv : moola.Function
                derivative
        """

        lb = self._lb
        ub = self._ub
        beta = self._beta
        tau = self._tau

        u_minus_w = g.copy()
        w = fenics.Function(self.function_space)

        g_vec = g.data.vector().get_local()
        u_vec = u.vector().get_local()

        w_vec = prox_box_l1(u_vec-(1/tau)*g_vec, lb, ub, beta/tau)
        w.vector().set_local(w_vec)
        u_minus_w.data.vector().set_local(u_vec - w_vec)

        psi = fw4pde.problem.ScaledL1Norm(u.function_space(),beta)

        psi_u = psi(u)
        psi_w = psi(w)

        result = deriv.apply(u_minus_w)
        result += psi_u - psi_w -.5*tau*fenics.norm(u_minus_w.data)**2

        return result


    def gap(self, u, g, deriv):

        lb = self._lb
        ub = self._ub
        beta = self._beta

        lmo = fw4pde.algorithms.MoolaBoxLMO(lb, ub, beta)

        w = g.copy()
        lmo.solve(g, w)

        u_minus_w = g.copy()
        u_minus_w.zero()

        u_minus_w.data.assign(u)
        u_minus_w.axpy(-1.0, w)

        psi = fw4pde.problem.ScaledL1Norm(u.function_space(),beta)

        psi_u = psi(u)
        psi_w = psi(w.data)

        dual_gap = deriv.apply(u_minus_w)
        dual_gap += psi_u - psi_w

        return dual_gap
