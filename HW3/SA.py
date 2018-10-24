import numpy as np
import random
import math
import time

DEBUG = 1
REAL_VALUE_MIN = -100
REAL_VALUE_MAX = 100


def random_accept(current_func_eval, new_function_eval, temperature):
    r = random.uniform(0, 1)
    p = math.e ** ((current_func_eval - new_function_eval) / temperature)
    return r < p


def is_allowed(candidate, allowed_list):

    for i in range(len(candidate)):
        if not allowed_list[i]:
            if not REAL_VALUE_MIN <= candidate[i] <= REAL_VALUE_MAX:
                return False

        elif not candidate[i] in allowed_list[i]:
            return False

    return True


def random_allowed_value(allowed_list):
    new_point = []
    for i in range(len(allowed_list)):
        new_point.append(random.choice(allowed_list) if allowed_list[i] else
                         random.uniform(REAL_VALUE_MIN, REAL_VALUE_MAX))
    return new_point


def multiplication_factor(a_i, c=2):
    if a_i > 0.6:
        return 1 + c * (a_i -0.6) / 0.4
    if a_i < 0.4:
        return 1 / (1 + c * (0.4 - a_i) / 0.4)
    return 1


def simulated_annealing(func, x0, allowed, t0, t_final, initial_step=1, step_reduction=0.9, temperature_reduction=0.5, num_iterations=5, num_cycles=2, time_constraint=15):
    input_dimension = len(x0)

    current_point = np.asarray(x0, dtype=float)
    current_func_eval = func(current_point)

    minimized_input_vector = current_point
    minimzed_func_eval = current_func_eval

    step_size = initial_step
    step_vector = np.array([initial_step] * input_dimension, dtype=float)

    acceptance_vector = np.array([1] * input_dimension, dtype=float)
    current_temperature = t0

    start = time.time()
    while current_temperature > t_final:
        if not DEBUG and time.time() - start > time_constraint:
            print("Warning: time expired. Temperature was not reached.")
            return minimized_input_vector, minimzed_func_eval

        for _ in range(num_iterations):
            for _ in range(num_cycles):
                for d in range(input_dimension):
                    random_step_weight = random.uniform(-1, 1)
                    new_step_point = current_point
                    new_step_point[d] += random_step_weight * step_vector[d]
                    if not is_allowed(new_step_point, allowed):
                        new_step_point = random_allowed_value(allowed)
                    new_step_function_val = func(new_step_point)

                    if new_step_function_val < current_func_eval:
                        current_func_eval = new_step_function_val
                        current_point = new_step_point
                    else:
                        if random_accept(current_func_eval, new_step_function_val, current_temperature):
                            current_func_eval = new_step_function_val
                            current_point = new_step_point
                        else:
                            acceptance_vector[d] -= 1.0 / num_cycles
                            acceptance_vector[d] = max(0, acceptance_vector[d])

                    if new_step_function_val < minimzed_func_eval:
                        minimized_input_vector = new_step_point
                        minimzed_func_eval = new_step_function_val

            for d in range(input_dimension):
                g = multiplication_factor(acceptance_vector[d])
                step_vector[d] *= g

        step_size *= step_reduction
        step_vector = np.array([step_size] * input_dimension)
        current_temperature *= temperature_reduction

    return minimized_input_vector, minimzed_func_eval


if __name__ == "__main__":
    test_func = lambda x: abs(x[0] - 3) + abs(x[1] - 5)
    print(simulated_annealing(test_func, (-10, -10), [[], []], 10, 0.1))