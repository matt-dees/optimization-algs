from ortools.algorithms import pywrapknapsack_solver
from test_functions import KnapsackProblem, GraphColoring
from SA import simulated_annealing
import time
from nelder_mead import nelder_mead
from scipy.optimize import fmin


def knapsack_verification():
    values = [360, 83, 59, 130, 431, 67, 230, 52, 93,
          125, 670, 892, 600, 38, 48, 147, 78, 256,
          63, 17, 120, 164, 432, 35, 92, 110, 22,
          42, 50, 323, 514, 28, 87, 73, 78, 15,
          26, 78, 210, 36, 85, 189, 274, 43, 33,
          10, 19, 389, 276, 312, 360, 83, 59, 130, 431, 67, 230, 52, 93,
          125, 670, 892, 600, 38, 48, 147, 78, 256,
          63, 17, 120, 164, 432, 35, 92, 110, 22,
          42, 50, 323, 514, 28, 87, 73, 78, 15,
          26, 78, 210, 36, 85, 189, 274, 43, 33,
          10, 19, 389, 276, 312, 360, 83, 59, 130, 431, 67, 230, 52, 93,
          125, 670, 892, 600, 38, 48, 147, 78, 256,
          63, 17, 120, 164, 432, 35, 92, 110, 22,
          42, 50, 323, 514, 28, 87, 73, 78, 15,
          26, 78, 210, 36, 85, 189, 274, 43, 33,
          10, 19, 389, 276, 312, 360, 83, 59, 130, 431, 67, 230, 52, 93,
          125, 670, 892, 600, 38, 48, 147, 78, 256,
          63, 17, 120, 164, 432, 35, 92, 110, 22,
          42, 50, 323, 514, 28, 87, 73, 78, 15,
          26, 78, 210, 36, 85, 189, 274, 43, 33,
          10, 19, 389, 276, 312]

    weights = [[7, 0, 30, 22, 80, 94, 11, 81, 70,
          64, 59, 18, 0, 36, 3, 8, 15, 42,
          9, 0, 42, 47, 52, 32, 26, 48, 55,
          6, 29, 84, 2, 4, 18, 56, 7, 29,
          93, 44, 71, 3, 86, 66, 31, 65, 0,
          79, 20, 65, 52, 13, 7, 0, 30, 22, 80, 94, 11, 81, 70,
          64, 59, 18, 0, 36, 3, 8, 15, 42,
          9, 0, 42, 47, 52, 32, 26, 48, 55,
          6, 29, 84, 2, 4, 18, 56, 7, 29,
          93, 44, 71, 3, 86, 66, 31, 65, 0,
          79, 20, 65, 52, 13, 7, 0, 30, 22, 80, 94, 11, 81, 70,
          64, 59, 18, 0, 36, 3, 8, 15, 42,
          9, 0, 42, 47, 52, 32, 26, 48, 55,
          6, 29, 84, 2, 4, 18, 56, 7, 29,
          93, 44, 71, 3, 86, 66, 31, 65, 0,
          79, 20, 65, 52, 13, 7, 0, 30, 22, 80, 94, 11, 81, 70,
          64, 59, 18, 0, 36, 3, 8, 15, 42,
          9, 0, 42, 47, 52, 32, 26, 48, 55,
          6, 29, 84, 2, 4, 18, 56, 7, 29,
          93, 44, 71, 3, 86, 66, 31, 65, 0,
          79, 20, 65, 52, 13]]

    capacities = [3400]

    custom_problem = KnapsackProblem(capacities[0], values, weights[0])
    start = time.time()
    custom_results = simulated_annealing(custom_problem.get_profit, [0] * len(values), [[0, 1] for _ in range(len(values))], 25000, 0.1)
    print(time.time() - start)

    print("Custom total value: ", custom_results[1])
    solver = pywrapknapsack_solver.KnapsackSolver(
        pywrapknapsack_solver.KnapsackSolver.
            KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
        'test')

    computed_value = solver.Solve()

    packed_items = [x for x in range(0, len(weights[0]))
                    if solver.BestSolutionContains(x)]
    packed_weights = [weights[0][i] for i in packed_items]
    total_weight = sum(packed_weights)

    print("Total value: ", computed_value)
    print("Total weight: ", total_weight)


def graph_coloring_verification():
    NUM_COLORS = 3
    NUM_NODES = 20
    COLOR_SET = range(NUM_COLORS)
    graph_color = GraphColoring(COLOR_SET, NUM_NODES)
    start = time.time()
    colors, _ = simulated_annealing(graph_color.solve, [0] * NUM_NODES, [COLOR_SET for _ in range(NUM_NODES)], 25000, 0.1)
    print(time.time() - start)
    graph_color.assign_color_map(colors)
    graph_color.print_colors()


def verify_nelder_mead():
    test_func = lambda x: x[0] ** 2 + 2 * x[1] ** 2 + 2 * x[0] * x[1]
    start = (100, 100)
    scipy_val = fmin(test_func, start)
    my_val = nelder_mead(test_func, start)
    print("SCIPY: ", scipy_val)
    print("MYVAL: ", my_val)
    return abs(my_val[1] - scipy_val[1]) <= 1e-2


if __name__ == "__main__":
    # graph_coloring_verification()
    verify_nelder_mead()