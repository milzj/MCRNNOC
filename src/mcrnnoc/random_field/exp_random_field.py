import numpy as np
from scipy import optimize
import itertools

class ExpRandomField(object):
    """Implements a random field

    Computes exp(KL) where KL is a truncated KL expansion.
    The truncated KL expansion is defined using the separable covariance operator
    considered in Example 7.56 in Lord, Powell, Shardlow (2014).

    Notes:
    (1) "num_addends" is actually not the number of addends in
    the KL expansion. The number of addends in this expansion
    is "(2*num_addends)**2."

    (2) function_space must be a nodal function space.

    (3) Code should not be run in parallel.

    (4) function space must consist of functions defined on (0,1)^2.
        Note: Example 7.56 considers eigenfunctions defined on
         (-a_1, a_1) x (-a_2, a_2). We transform (0,1)^2 to
        (-1/2,1/2)^2 using the mapping y to y-1/2.


    References:
    ----------

    G. J. Lord, C. E. Powell, and T. Shardlow, An Introduction to Computational
    Stochastic PDEs, Cambridge Texts Appl. Math. 50, Cambridge University Press, Cam-
    bridge, 2014, https://doi.org/10.1017/CBO9781139017329

    """

    def __init__(self, function_space, options_random_field):
        # cos cos, sin cos, cos sin, sin sin

        options_rf = options_random_field

        self.len_scale = options_rf["len_scale"]
        num_addends = options_rf["num_addends"]
        self.num_addends = num_addends
        self.function_space = function_space

        self.odd_list = np.arange(0, num_addends)[1::2]
        self.even_list = np.arange(0, num_addends)[0::2]
        self.a = 0.5

        self._num_rvs = num_addends**2

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
        odd_roots = np.zeros(len(self.odd_list))
        even_roots = np.zeros(len(self.even_list))

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

    def compute_addends(self):
        """Compute addends of the KL expansion.

        We compute the addends of the 1D KL expansion
        as described in Example 7.55.
        """

        a = self.a
        len_scale = self.len_scale
        num_addends = self.num_addends
        amplitudes = []
        frequencies = []

        # odd
        k = 0
        for i in self.odd_list:
            omega = np.pi*self.odd_roots[k]
            frequencies.append(omega)
            A = 1.0/np.sqrt(a+np.sin(2*omega*a)/2/omega)
            nu = 2.0/len_scale/(omega**2+1/len_scale**2)
            amplitudes.append(A*np.sqrt(nu))
            k+=1

        # even
        k = 0
        for i in self.even_list:
            omega = np.pi*self.even_roots[k]
            frequencies.append(omega)
            B = 1.0/np.sqrt(a-np.sin(2*omega*a)/2/omega)
            nu = 2.0/len_scale/(omega**2+1/len_scale**2)
            amplitudes.append(B*np.sqrt(nu))
            k+=1

        self.amplitudes = amplitudes
        self.frequencies = frequencies


    def eigenfunctions(self, i, k):

        if i % 2 != 0 and k % 2 != 0:
            A_i = "1.0/sqrt(a+sin(2*omega_i*a)/2/omega_i)"
            A_k = "1.0/sqrt(a+sin(2*omega_k*a)/2/omega_k)"
            return A_i+"*"+"cos(omega_i*(x[0]-a))"+"*"+A_k+"*"+"cos(omega_k*(x[1]-a))"
        elif i % 2 != 0 and k % 2 == 0:
            A_i = "1.0/sqrt(a+sin(2*omega_i*a)/2/omega_i)"
            B_k = "1.0/sqrt(a-sin(2*omega_k*a)/2/omega_k)"
            return A_i+"*"+"cos(omega_i*(x[0]-a))"+"*"+B_k+"*"+"sin(omega_k*(x[1]-a))"
        elif i % 2 == 0 and k % 2 != 0:
            B_i = "1.0/sqrt(a-sin(2*omega_i*a)/2/omega_i)"
            A_k = "1.0/sqrt(a+sin(2*omega_k*a)/2/omega_k)"
            return B_i+"*"+"sin(omega_i*(x[0]-a))"+"*"+A_k+"*"+"cos(omega_k*(x[1]-a))"
        elif i % 2 == 0 and k % 2 == 0:
            B_i = "1.0/sqrt(a-sin(2*omega_i*a)/2/omega_i)"
            B_k = "1.0/sqrt(a-sin(2*omega_k*a)/2/omega_k)"
            return B_i+"*"+"sin(omega_i*(x[0]-a))"+"*"+B_k+"*"+"sin(omega_k*(x[1]-a))"

    def omegas(self, i):

        j = int(np.ceil(i/2))

        if i % 2 != 0:
            return np.pi*self.odd_roots[j-1]
        else:
            return np.pi*self.even_roots[j-1]

    def eigenvalues(self, l, omega):

        return 2/l/(1/l**2+omega**2)

    def compute_2d_addends(self):
        """Compute addends of the KL expansion.

        We compute the addends of the 2D KL expansion
        as described in Example 7.56, that is, all combinations
        of certain 1D KL expansions.

        TODO: Improve current implementation.
        """

        import fenics

        function_space = self.function_space
        element = function_space.ufl_element()
        mpi_comm = function_space.mesh().mpi_comm()
        v = fenics.Function(function_space)
        self.w  = fenics.Function(function_space)

        len_scale = self.len_scale
        a = self.a

        num_addends = self.num_addends

        _addends = []

        idx_pairs = list(itertools.product(range(1, self.num_addends+1), repeat=2))

        for (i,k) in idx_pairs:
            expression_str = self.eigenfunctions(i, k)
            omega_i = self.omegas(i)
            omega_k = self.omegas(k)
            eigenvalue_i = self.eigenvalues(len_scale, omega_i)
            eigenvalue_k = self.eigenvalues(len_scale, omega_k)
            s = np.sqrt(eigenvalue_i)*np.sqrt(eigenvalue_k)
            eigenfunction_ = fenics.Expression("s*"+expression_str, s = s, a = a, \
                                    omega_i = omega_i, omega_k = omega_k, element=element, mpi_comm = mpi_comm)
            v.interpolate(eigenfunction_)
            _addends.append(v.vector().get_local())

        self.addends = np.vstack(_addends).T

    def sample_vec(self, samples):
        """Compute a sample of KL expansion.

        We compute the sample using a matrix vector multiplication
        with vector being the samples.
        """

        addends = self.addends
        field = addends @ samples

        return np.exp(field)

    def sample(self, samples):
        """Compute a sample of KL expansion.

        We compute the sample using a matrix vector multiplication
        with vector being the samples.
        """

        w = self.w
        sample_vec = self.sample_vec(samples)
        w.vector().set_local(sample_vec)

        return w



    def __call__(self, samples):
        return self.sample(samples)
