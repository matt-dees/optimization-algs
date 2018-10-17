from optimizer_1d import optimizer1D
import numpy
import math


EPSILON_MACHINE = 1.11e-16
EPSILON_ABSOLUTE = 1.11e-14
EPSILON_RELATIVE = math.sqrt(EPSILON_MACHINE)
EPSILON_GRADIENT = EPSILON_ABSOLUTE
MINIMIZER_STEP_SIZE = 1


def check_gradient_stop(gradient):
    return numpy.linalg.norm(gradient) <= EPSILON_GRADIENT

def check_func_eval_stop(fx0, fx1):
    return abs(fx1 - fx0) <= EPSILON_ABSOLUTE + EPSILON_RELATIVE * abs(fx0)

def take_steepest_descent_step(func, x0, fprime):
    x0 = numpy.asarray(x0)
    f0 = numpy.asarray(func(x0))
    gradient_x0 = numpy.asarray(fprime(x0))
    d0 = -1 * gradient_x0
    falpha = lambda alpha: func(x0 + (alpha * d0))
    alpha = optimizer1D(falpha, x0[0], MINIMIZER_STEP_SIZE)
    x1 = x0 + alpha * d0
    return x1, d0

def take_conjugate_gradient_step(func, x0, d0, x1, fprime):
    gradient_x1 = numpy.asarray(fprime(x1))
    gradient_x0 = numpy.asarray(fprime(x0))
    beta = numpy.linalg.norm(gradient_x1) ** 2 / numpy.linalg.norm(gradient_x0) ** 2
    d1 = -gradient_x1 + beta * d0
    falpha = lambda alpha: func(x1 + (alpha * d1))
    alpha = optimizer1D(falpha, x1[0], MINIMIZER_STEP_SIZE)
    return x1 + alpha * d1, d1

def conjugate_gradients(func, x0, fprime, restart_frequency):
    cg_iterations = 0
    d1 = None
    x1 = None

    while True:
        if cg_iterations == 0 or cg_iterations == restart_frequency:
            x1, d0 = take_steepest_descent_step(func, x0, fprime)
        else:
            x1, d1 = take_conjugate_gradient_step(func, x0, d0, x1, fprime)
        cg_iterations += 1
        if check_gradient_stop(fprime(x1)) or check_func_eval_stop(func(x0), func(x1)):
            return(x1, func(x1), cg_iterations)

        if d1 is not None:
            d0 = d1
        x0 = x1

test_func = lambda x: x[0] ** 2 + (x[1] - 2) ** 2
fprime = lambda x: (2*x[0], 2*(x[1] - 2))
print(conjugate_gradients(test_func, (.1, .1), fprime, 10))