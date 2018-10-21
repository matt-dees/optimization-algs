from optimizer_1d import optimizer1D
import numpy
import math
import time
from collections import namedtuple

EPSILON_MACHINE = 1.11e-16
EPSILON_ABSOLUTE = 1.11e-14
EPSILON_RELATIVE = math.sqrt(EPSILON_MACHINE)
EPSILON_GRADIENT = EPSILON_ABSOLUTE
MINIMIZER_STEP_SIZE = 1

GCOptOutput = namedtuple("GCOptOutput", "minimizing_input_vector minimized_output gc_iterations")


def check_gradient_stop(gradient):
    return numpy.linalg.norm(gradient) <= EPSILON_GRADIENT


def check_func_eval_stop(fx0, fx1):
    return abs(fx1 - fx0) <= EPSILON_ABSOLUTE + EPSILON_RELATIVE * abs(fx0)


def take_steepest_descent_step(func, x0, fprime):
    x0 = numpy.asarray(x0, dtype=numpy.float64)
    f0 = numpy.asarray(func(x0), dtype=numpy.float64)
    gradient_x0 = numpy.asarray(fprime(x0), dtype=numpy.float64)
    d0 = -1 * gradient_x0
    d0_magnitude = numpy.linalg.norm(d0)
    d0 /= d0_magnitude
    falpha = lambda alpha: func(x0 + (alpha * d0))
    alpha = optimizer1D(falpha, x0[0], MINIMIZER_STEP_SIZE)
    x1 = x0 + alpha * d0
    return x1, d0


def take_conjugate_gradient_step(func, x0, d0, x1, fprime):
    gradient_x1 = numpy.asarray(fprime(x1), dtype=numpy.float64)
    gradient_x0 = numpy.asarray(fprime(x0), dtype=numpy.float64)
    beta = numpy.linalg.norm(gradient_x1) ** 2 / numpy.linalg.norm(gradient_x0) ** 2
    d1 = -gradient_x1 + beta * d0
    d1 /= numpy.linalg.norm(d1)
    falpha = lambda alpha: func(x1 + (alpha * d1))
    alpha = optimizer1D(falpha, x1[0], MINIMIZER_STEP_SIZE)
    return x1 + alpha * d1, d1

def alternate_method(func, x0, fprime, epsilon_abs=EPSILON_ABSOLUTE, time_constraint=15):
    return steepest_descent(func, x0, fprime, epsilon_abs, time_constraint)

def steepest_descent(func, x0, fprime, epsilon_abs=EPSILON_ABSOLUTE, time_constraint=15):
    global EPSILON_ABSOLUTE
    EPSILON_ABSOLUTE = epsilon_abs

    f0 = func(x0)
    start = time.time()
    should_stop = 0
    while True:
        x1, _ = take_steepest_descent_step(func, x0, fprime)
        f1 = func(x1)

        if check_gradient_stop(fprime(x1)) or check_func_eval_stop(f0, f1):
            if should_stop:
                return [x1, func(x1)]
            else:
                should_stop = 1
        else:
            should_stop = 0

        if time.time() - start >= time_constraint:
            print("Warning: optimizer ran for longer than time contraint. No stopping criteria was met.")
            return [x1, func(x1)]

        x0, f0 = x1, f1


def conjugate_gradients(func, x0, fprime, restart_frequency, epsilon_abs=EPSILON_ABSOLUTE, time_constraint=15):
    global EPSILON_ABSOLUTE
    EPSILON_ABSOLUTE = epsilon_abs

    cg_iterations = 0
    d1 = None
    x1 = None
    should_stop = False

    start = time.time()

    while True:
        if cg_iterations == 0 or cg_iterations % restart_frequency == 0:
            x1, d0 = take_steepest_descent_step(func, x0, fprime)
            d1 = None
        else:
            x1, d1 = take_conjugate_gradient_step(func, x0, d0, x1, fprime)
        cg_iterations += 1
        if check_gradient_stop(fprime(x1)) or check_func_eval_stop(func(x0), func(x1)):
            if should_stop:
                return [x1, func(x1), cg_iterations]
            else:
                should_stop = 1
        else:
            should_stop = 0

        if time.time() - start >= time_constraint:
            print("Warning time expired. No minimum found via stopping critera.")
            return [x1, func(x1), cg_iterations]

        if d1 is not None:
            d0 = d1
        x0 = x1


if __name__ == "__main__":
    test_func = lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    fprime = lambda x: (2*(200 * x[0] ** 3 - 200 * x[0] * x[1] + x[0] - 1), #partial x0
                        200*(x[1] - x[0]**2))                               #partial x1
    print(conjugate_gradients(test_func, (9, 9), fprime, 10))
    print(steepest_descent(test_func, (9, 9), fprime, 10))

    easy_func = lambda x: (1 - x[0]) ** 4 + (2 - x[1]) ** 4
    easy_dunc_prime = lambda x: (-4 * (1 - x[0]) ** 3,
                                 -4 * (2 - x[1]) ** 3)

    print(conjugate_gradients(easy_func, (7, 7), easy_dunc_prime, 10))
    print(steepest_descent(easy_func, (.1, .1), easy_dunc_prime, 10))

    test_func = lambda x: x[0] * x[1] + sum([(ele - x[ele - 1]) ** 2 for ele in range(1, 11)])
    fprime = lambda x: [x[1] + -2 * (1 - x[0]),
                        x[0] + -2 * (2 - x[1])] + \
                        [-2*(ele - x[ele - 1]) for ele in range(3, 11)]

    print(conjugate_gradients(test_func, [.23] * 10, fprime, 100))
    print(steepest_descent(test_func, [.1] * 10, fprime, 10))