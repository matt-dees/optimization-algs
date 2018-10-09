from function_test_harness import TestHarness
from optimizer_1d import Optimizer1D
from optimizer_2d import Optimizer2D
from stats_generator import StatsGenerator
from graphic_generator import GraphicGenerator

import random
from collections import namedtuple
import matplotlib.pyplot as plt

FunctionUnderTest = namedtuple("FunctionUnderTest", "func func_name optimum")

NUM_RUNS = 1000


def test_functions(optimizer, funcs_to_test, start_params, step_params):
    for func in funcs_to_test:
        report = TestHarness.test_optimizer(optimizer, func.func, start_params, step_params)
        stats.append(StatsGenerator.generate_stats(run_report_list=report, optimum=func.optimum))
        reports.append(report)

    for i, func in enumerate(funcs_to_test):
        GraphicGenerator.generate_box_plots(func, reports[i])

    GraphicGenerator.generate_stats_table(funcs_to_test, stats)


if __name__ == "__main__":

    reports = []
    stats = []
    funcs_to_test = [FunctionUnderTest(func=lambda x: (x - 2) ** 2, optimum=2, func_name="(x-2)^2"),
                     FunctionUnderTest(func=lambda x: abs(x - 5), optimum=5, func_name="abs(x-5)")]

    # Random distribution of start parameter
    start = [random.randint(-10000, 10000) for x in range(NUM_RUNS)]

    # Random distribution for the inital_step parameter
    step = [1 for x in range(NUM_RUNS)]

    test_functions(Optimizer1D.golden_section, funcs_to_test, start, step)

    funcs_to_test_2d = [FunctionUnderTest(func=lambda x, y: 5*(x - 2.5)**2- 6*(x - 2.5)*(y - 20.5) + 5*(y - 20.5)**2, optimum=(2.5, 20.5), func_name="5(x-2.5)^2 - 6(x-2.5)(y-20.5) + 5(y-20.5)^2"),
                        FunctionUnderTest(func=lambda x, y: abs(x + 2.0) + abs(y - 1.0), optimum=(-2, 1), func_name="abs(x+2.0) + abs(y-1.0)")]

    # Random distribution of start parameter
    start = [(random.randint(-10000, 10000), random.randint(-10000, 10000)) for x in range(NUM_RUNS)]

    # Random distribution for the inital_step parameter
    step = [(1, 1) for x in range(NUM_RUNS)]

    test_functions(Optimizer2D.coordinate_descent, funcs_to_test_2d, start, step)


