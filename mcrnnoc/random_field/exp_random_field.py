import numpy as np
from scipy import optimize
import itertools

from mcrnnoc.random_field.options_random_field import OptionsRandomField

from fenics import *
from dolfin_adjoint import *

class ExpRandomField(object):
    """Implements a random field

    Computes exp(KL) where KL is a truncated KL expansion.
    The truncated KL expansion is defined using the separable covariance operator
    considered in Example 7.56 in Lord, Powell, Shardlow (2014).

    Note: "num_addends" is actually not the number of addends in
    the KL expansion. The number of addends in this expansion
    is "(2*num_addends)**2."

    function_space must be a nodal function space.

    References:
    ----------

    G. J. Lord, C. E. Powell, and T. Shardlow, An Introduction to Computational
    Stochastic PDEs, Cambridge Texts Appl. Math. 50, Cambridge University Press, Cam-
    bridge, 2014, https://doi.org/10.1017/CBO9781139017329

    """

    def __init__(self, function_space):
        # cos cos, sin cos, cos sin, sin sin

        options_rf = OptionsRandomField().options

        self.len_scale = options_rf["len_scale"]
        num_addends = options_rf["num_addends"]
        self.num_addends = num_addends
        self.function_space = function_space

        self.odd_list = np.arange(0, 2*num_addends)[1::2]
        self.even_list = np.arange(0, 2*num_addends)[0::2]
        self.a = 0.5

        self._num_rvs = (2*num_addends)**2

        self.compute_roots()
        self.compute_addends()
        self.compute_2d_addends()

    @property
    def num_rvs(self):
        "Number of random variables in the KL expansion."
        return self._num_rvs

    def compute_roots(self):
        """Computes the roots of the functions fodd and feven.

        Since tan appears in the definition of fodd and feven,
        we multiply fodd and feven with cos and subsequently
        apply Brent's method to compute roots. After root computations,
        we perform a simple test whether the computations
        were successful. The multiplication by cos tries to
        simplify root finding. Furthermore we perform the
        variable transformation pi*x = omega. The roots of
        the transformed functions appear to be close to
        natural numbers (empirical finding via WolframAlpha).

        See page 295 in Lord, Powell, Shardlow.

        """

        len_scale = self.len_scale
        a = self.a

        num_addends = self.num_addends
        odd_roots = np.zeros(num_addends)
        even_roots = np.zeros(num_addends)

        #fodd transformed by multiplying by cos(pi*x*a)
        fodd = lambda x: 1.0/len_scale*np.cos(np.pi*x*a) - np.pi*x*np.sin(np.pi*x*a)
        k = 0
        for i in self.even_list:
            if i == 0:
                root = optimize.brentq(fodd, 0, i+1)
            else:
                root = optimize.brentq(fodd, i, i+1)
            odd_roots[k] = root
            k += 1

        self.odd_roots = odd_roots

        #feven 1/l tan(xa) + x = 0 <=> 1/l sin(xa) + x cos(xa) = 0
        feven = lambda x: 1.0/len_scale*np.sin(np.pi*x*a) + np.pi*x*np.cos(np.pi*x*a)
        k = 0
        for i in self.odd_list:
            if i == 0:
                root = optimize.brentq(feven, 0, i+1)
            else:
                root = optimize.brentq(feven, i, i+1)
            even_roots[k] = root
            k += 1

        self.even_roots = even_roots

        # Tests
        atol = 1e-3
        fodd = lambda x: 1.0/len_scale-np.pi*x*np.tan(np.pi*x*a)
        for x in odd_roots:
            assert fodd(x) < atol, "Error in root computation for fodd"

        assert np.all(np.diff(odd_roots) > 0.0) == True

        feven = lambda x: 1.0/len_scale*np.tan(np.pi*x*a) + np.pi*x
        for x in even_roots:
            assert feven(x) < atol, "Error in root computation for feven"

        assert np.all(np.diff(even_roots) > 0.0) == True

    @profile
    def compute_addends(self):
        """Compute addends of the KL expansion.

        We compute the addends of the 1D KL expansion
        as described in Example 7.55.

        """

        a = self.a
        len_scale = self.len_scale
        num_addends = self.num_addends
        eigenfunctions_times_sqrt = []

        # odd
        k = 0
        for i in self.odd_list:
            omega = np.pi*self.odd_roots[k]
            A = 1.0/np.sqrt(a+np.sin(2*omega*a)/2/omega)
            nu = 2.0/len_scale/(omega**2+1/len_scale**2)
            eigenfunctions_times_sqrt.append("{}*cos({}*x)".format(A*np.sqrt(nu), omega))
            k+=1

        # even
        k = 0
        for i in self.even_list:
            omega = np.pi*self.even_roots[k]
            B = 1.0/np.sqrt(a-np.sin(2*omega*a)/2/omega)
            nu = 2.0/len_scale/(omega**2+1/len_scale**2)
            eigenfunctions_times_sqrt.append("{}*sin({}*x)".format(B*np.sqrt(nu), omega))
            k+=1

        self.eigenfunctions_times_sqrt = eigenfunctions_times_sqrt

    @profile
    def compute_2d_addends(self):
        """Compute addends of the KL expansion.

        We compute the addends of the 2D KL expansion
        as described in Example 7.56.
        """

        eigenfunctions_times_sqrt = self.eigenfunctions_times_sqrt

        products = list(itertools.product(eigenfunctions_times_sqrt, eigenfunctions_times_sqrt))

        addends = []

        v = Function(self.function_space)

        for i in products:
            a, b = i
            a = a.replace("x", "(x[0]-0.5)")
            b = b.replace("x", "(x[1]-0.5)")
            c = "{}*{}".format(a,b)
            v_expr = Expression(c, degree=1)
            v.interpolate(v_expr)
            addends.append(v.vector().get_local())

        # Convert list to matrix
        self.addends = np.vstack(addends).T

    @profile
    def sample(self, samples):
        """Compute a sample of KL expansion."""

        addends = self.addends
        w = Function(self.function_space)

        field = addends @ samples
        w.vector()[:] = np.exp(field)

        return w

