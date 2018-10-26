import numpy as np
import time

EPSILON_ABSOLUTE = 1e-14
EPSILON_RELATIVE = 1e-12


def create_simplex(point, func, c):
    simplex = [(point, func(point))]
    for i in range(len(point)):
        new_point = point.copy()
        new_point[i] +=c
        simplex.append((new_point, func(new_point)))

    simplex.sort(key=lambda x_f_pair: x_f_pair[1])
    return simplex


def stopping_criteria(f_new, f_old):
    return abs(f_new - f_old) < EPSILON_ABSOLUTE + EPSILON_RELATIVE * abs(f_old)


def reflection(x_b, x_h, r=1):
    return x_b + r * (x_b - x_h)


def expansion(x_r, x_b, e=1):
    return x_r + e * (x_r - x_b)


def contraction(x_b, x_h, c=0.5):
    return x_b + c * (x_h - x_b)


def scale(x, x_l, s=0.5):
    return x_l + s * (x - x_l)


def nelder_mead(func, initial_point, c=1, step_reduction=0.9, time_constraint=15):
    current_point = np.asarray(initial_point)
    x_f_min = current_point, func(current_point)

    iteration_count = 0
    simplex = create_simplex(current_point, func, c)
    start = time.time()

    while True:
        if time.time() - start >= time_constraint:
            print("Warning: time expired.")
            return x_f_min
        iteration_count += 1
        simplex.sort(key=lambda x_f_pair: x_f_pair[1])
        x_f_h = simplex[-1]
        x_f_s = simplex[-2]
        x_f_l = simplex[0]
        if x_f_l[1] < x_f_min[1]:
            x_f_min = x_f_l
        if stopping_criteria(x_f_h[1], x_f_l[1]):
            if iteration_count >= 1 and stopping_criteria(x_f_min[1], x_f_l[1]):
                return x_f_min
            x_f_min = x_f_l
            c *= step_reduction
            simplex = create_simplex(x_f_l[0], func, c)
            continue

        x_b = 1.0 / len(simplex[:-1]) * sum(list(map(lambda x: x[0], simplex[:-1])))

        x_h = np.asarray(x_f_h[0], float)
        x_r = reflection(x_b, x_h)

        x_f_r = x_r, func(x_r)

        if x_f_r[1] >= x_f_l[1]:
            if x_f_r[1] <= x_f_h[1]:
                x_f_h = x_f_r
            if x_f_r[1] <= x_f_s[1]:
                simplex = [x_f_l, x_f_s, x_f_h]
                continue
            x_c = contraction(x_b, x_f_h[0])
            x_f_c = x_c, func(x_c)
            if x_f_c[1] <= x_f_h[1]:
                x_f_h = x_f_c
            x_l = scale(x_f_l[0], x_f_l[0])
            x_s = scale(x_f_s[0], x_f_l[0])
            x_h = scale(x_f_h[0], x_f_l[0])

            x_f_l = x_l, func(x_l)
            x_f_s = x_s, func(x_s)
            x_f_h = x_h, func(x_h)
        else:
            x_e = expansion(x_r, x_b)
            x_f_e = (x_e, func(x_e))

            if x_f_e[1] >= x_f_l[1]:
                x_f_h = x_f_e
            else:
                x_f_h = x_f_r

        simplex = [x_f_l, x_f_s, x_f_h]


if __name__ == "__main__":
    #nm_test_function = lambda x: x[0] + x[1]
    nm_test_function_2 = lambda x: x[0] ** 2 + 2 * x[1] ** 2 + 2 * x[0] * x[1]
    #print(nelder_mead(nm_test_function, (0, 1)))
    print(nelder_mead(nm_test_function_2, (0, 1)))