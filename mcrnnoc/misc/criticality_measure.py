
import fenics
from mcrnnoc.prox import prox_box_l1

def criticality_measure(control, gradient, lb, ub, beta):
    """Evaluated a "canonical" criticality measure.

    Computes the L2-norm of

    control - prox(control - gradient)

    Note: This function should not be used within parallel
    computations.

    """

    control_vec = control.vector()[:]
    gradient_vec = gradient.vector()[:]
    projected_gradient = fenics.Function(control.function_space())


    x_vec = control_vec - gradient_vec
    w_vec = prox_box_l1(x_vec, lb, ub, beta)
    projected_gradient.vector()[:] = w_vec

    return fenics.errornorm(control, projected_gradient, degree_rise = 0)

