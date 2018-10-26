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
        new_point.append(random.choice(allowed_list[i]) if allowed_list[i] else
                         random.uniform(REAL_VALUE_MIN, REAL_VALUE_MAX))
    return np.asarray(new_point)


def multiplication_factor(a_i, c=2):
    if a_i > 0.6:
        return 1 + c * (a_i - 0.6) / 0.4
    if a_i < 0.4:
        return 1 / (1 + c * (0.4 - a_i) / 0.4)
    return 1


def create_unit_vector(size, index):
    unit_vec = np.zeros(size)
    unit_vec[index] = 1
    return unit_vec


def get_new_point(current_point, step, points_allowed):
    if not points_allowed:
        random_step_weight = random.uniform(-1, 1)
        return current_point + random_step_weight * step

    return random.choice(list(filter(lambda x: x != current_point, points_allowed)))


def simulated_annealing(func, x0, allowed, t0, t_final, initial_step=10, step_reduction=0.9, temperature_reduction=0.5, num_iterations=5, num_cycles=20, time_constraint=15):
    input_dimension = len(x0)

    current_point = np.asarray(x0, dtype=np.float64)
    current_func_eval = func(current_point)

    minimized_input_vector = current_point
    minimized_func_eval = current_func_eval

    step_size = initial_step
    step_vector = np.array([initial_step] * input_dimension, dtype=np.float64)

    acceptance_vector = np.array([1] * input_dimension, dtype=np.float64)
    current_temperature = t0

    start = time.time()
    while current_temperature > t_final:
        if not DEBUG and time.time() - start > time_constraint:
            print("Warning: time expired. Temperature was not reached.")
            return minimized_input_vector, minimized_func_eval

        for _ in range(num_iterations):
            for _ in range(num_cycles):
                for d in range(input_dimension):

                    new_step_point = current_point.copy()
                    new_step_point[d] = np.float64(get_new_point(current_point[d], step_vector[d], allowed[d]))
                    if not is_allowed(new_step_point, allowed):
                        new_step_point = random_allowed_value(allowed)
                    if type(new_step_point[0]) is list:
                        return
                    new_step_function_val = func(new_step_point)

                    if new_step_function_val < current_func_eval:
                        current_func_eval = new_step_function_val
                        current_point = new_step_point
                        if current_func_eval < minimized_func_eval:
                            minimized_input_vector = current_point
                            minimized_func_eval = current_func_eval
                    else:
                        if random_accept(current_func_eval, new_step_function_val, current_temperature):
                            current_func_eval = new_step_function_val
                            current_point = new_step_point
                        else:
                            acceptance_vector[d] -= 1.0 / num_cycles
                            acceptance_vector[d] = max(0, acceptance_vector[d])

            for d in range(input_dimension):
                g = multiplication_factor(acceptance_vector[d])
                step_vector[d] *= g

        step_size *= step_reduction
        step_vector = np.array([step_size] * input_dimension, dtype=np.float64)
        current_temperature *= temperature_reduction

    return list(minimized_input_vector), minimized_func_eval


if __name__ == '__main__':
    items = [(10.0, 3.3),(1,.001),(1,.005),(1,5.0),(3.0,.01),(5.0,3.0)]
    def testf(x):
        max_weight = 10.0
        total_profit = -1.0*sum([x[i]*items[i][0] for i in range(len(items))])
        total_weight = sum([x[i]*items[i][1] for i in range(len(items))])
        if total_weight > max_weight or x[-1] < 0.05:
            return 1.0e12
        else:
            return total_profit/x[-1]
    allowed = [[0,1] for i in range(len(items))] + [[]]
    starting_temp = 10
    ending_temp = 0.1
    print(simulated_annealing(testf, [0,0,0,0,0,0,1.5], allowed, starting_temp, ending_temp))