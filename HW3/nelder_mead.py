import numpy as np

EPSILON_ABSOLUTE = 1e6
EPSILON_RELATIVE = 1e4


def create_simplex(point, func, c):
    simplex = [(point, func(point))]
    for i in range(len(point)):
        new_point = point.copy()
        new_point[i] +=c
        simplex.append((point, func(point)))

    return simplex


def stopping_criteria(f_new, f_old):
    return f_new - f_old < EPSILON_ABSOLUTE + EPSILON_RELATIVE * abs(f_old)

def reflection(x_b, x_h, r=1):
    return x_b + r * (x_b - x_h)

def nelder_mead(func, initial_point, c=1):
    current_point = initial_point
    current_func_val = func(initial_point)

    while True:
        iteration_count = 0
        simplex = create_simplex(current_point, c)
        simplex.sort(key=lambda x_f_pair: x_f_pair[1])

        x_f_h = simplex[-1]
        x_f_s = simplex[-2]
        x_f_l = simplex[0]
        if stopping_criteria(x_f_h[1], x_f_l[1]):
            break

        x_b = np.mean(list(map(lambda x: x[0], simplex)))

        x_b = np.asarray(x_b)
        x_h = np.asarray(x_f_h[0])
        x_r = reflection(x_b, x_h)

        x_f_r = (x_r, func(x_r))

        
