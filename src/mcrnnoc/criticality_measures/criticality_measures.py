from mcrnnoc.criticality_measures.prox import prox_box_l1
import numpy as np

class CriticalityMeasures(object):
    """Evaluates several criticality measures

    Computes the criticality measures for the optimization problem

    min_x g^T x + beta*norm(x,1) subject to lb <= x <= ub.

    The proximal operator is computed using a composition formula.

    Parameters:
    -----------
        x : ndarray or float
            input array
        lb, ub : ndarray or float
            lower and upper bounds
        g : ndarray or float
            gradient evaluated at x
        beta : float
            regularization parameter, nonnegative
        tau : float
            criticality measure parameter, positive
    """

    def __init__(self,lb,ub,beta,tau=1.0):

        self._lb = lb
        self._ub = ub
        self._beta = beta
        self._tau = tau

    def prox(self,v):

        lb = self._lb
        ub = self._ub
        beta = self._beta
        tau = self._tau

        return prox_box_l1(v, lb, ub, beta/tau)

    def proj(self,v):

        lb = self._lb
        ub = self._ub

        return prox_box_l1(v, lb, ub, 0.0)

    def canonical_residual(self, x, g):
        """Evaluated the canonical residual

        x - prox_{\psi/tau}(x-(1/tau)*g(x))

        """

        tau = self._tau
        prox_v = self.prox(x-(1/tau)*g)
        self._canonical_residual = x - prox_v

    def canonical_map(self, x, g):
        """Computes the 2-norm of the canonical residual."""
        self.canonical_residual(x,g)
        return np.linalg.norm(self._canonical_residual)

    def normal_residual(self, v, g):
        """Evaluated the normal residual

        tau*(v-prox(v)) + g(prox(v)).

        If v is not supplied, the function computes
        v according to v = x - (1/tau)*g and assumes that
        g(x) = g(prox(v)).
        """

        tau = self._tau
        prox_v = self.prox(v)
        self._normal_residual = tau*(v-prox_v)+g

    def normal_map(self, v, g):
        """Computes the 2-norm of the normal residual."""
        self.normal_residual(v,g)
        return np.linalg.norm(self._normal_residual)

    def rgap(self,x,g):
        """Evaluated the regularized gap function

        (g,x-w)_H + \psi(x)-\psi(w)-(nu/2)\|x-w\|_H^2,

        where w = prox_{\psi/tau}(x-(1/tau)*g) and \psi(u) = beta*\|u\|_1
        """
        lb = self._lb
        ub = self._ub
        beta = self._beta
        tau = self._tau

        prox_v = prox_box_l1(x-(1/tau)*g, lb, ub, beta/tau)
        psi_x = np.linalg.norm(x,1)
        psi_prox_v = np.linalg.norm(prox_v,1)
        result = g@(x-prox_v) + beta*psi_x - beta*psi_prox_v - tau/2*np.linalg.norm(x-prox_v)**2
        return result
