from function_test_harness import TestHarness
from optimizer_1d import Optimizer1D
from stats_generator import StatsGenerator
from graphic_generator import GraphicGenerator

import random
from collections import namedtuple
import matplotlib.pyplot as plt

FunctionUnderTest = namedtuple("FunctionUnderTest", "func func_name optimum")

NUM_RUNS = 1000

if __name__ == "__main__":

    reports = []
    stats = []
    funcs_to_test = [FunctionUnderTest(func=lambda x: (x - 2) ** 2, optimum=2, func_name="(x-2)^2"),
                     FunctionUnderTest(func=lambda x: abs(x - 5), optimum=5, func_name="abs(x-5)")]

    test_harness = TestHarness()
    test_harness.load_optimizer(Optimizer1D(Optimizer1D.golden_section))

    # Random distribution of start parameter
    start = [random.randint(-10000, 10000) for x in range(NUM_RUNS)]

    # Random distribution for the inital_step parameter
    step = [1 for x in range(NUM_RUNS)]

    for func in funcs_to_test:
        test_harness.load_test_function(func.func)
        report = test_harness.test_optimizer(start, step)
        stats.append(StatsGenerator.generate_stats(run_report_list=report, optimum=func.optimum))
        reports.append(report)

    for i, func in enumerate(funcs_to_test):
        GraphicGenerator.generate_box_plots(func, reports[i])

    GraphicGenerator.generate_stats_table(funcs_to_test, stats)

