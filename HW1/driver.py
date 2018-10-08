from function_test_harness import TestHarness
from optimizer_1d_blackbox import Optimizer1D
import random
from stats_generator import StatsGenerator
from collections import namedtuple
import matplotlib.pyplot as plt

FunctionUnderTest = namedtuple("FunctionUnderTest", "func func_name optimum")
NUM_FUNC_EVALS_STRING = "Number of function evaluations"
TIME_ELAPSED_STRING = "Time elapsed (ms)"
RELATIVE_DISTANCE_STRING = "Relative Distance from Optimum (%)"

if __name__ == "__main__":

    reports = []
    stats = []
    funcs_to_test = [FunctionUnderTest(func=lambda x: (x - 2) ** 2, optimum=2, func_name="(x-2)^2"),
                     FunctionUnderTest(func=lambda x: abs(x - 5), optimum=5, func_name="abs(x-5)")]

    test_harness = TestHarness()
    test_harness.load_optimizer(Optimizer1D(Optimizer1D.golden_section))

    # Random distribution of start parameter
    start = random.sample(range(-10000, 10000), 1000)

    # Step always equals 1 for the 1000 runs
    step = [1] * len(start)

    for func in funcs_to_test:
        test_harness.load_test_function(func.func)
        report = test_harness.test_optimizer(start, step)
        stats.append(StatsGenerator.generate_stats(run_report_list=report, optimum=func.optimum, function_name=func.func_name))
        reports.append(report)

    cell_text = []
    for stat in stats:
        cell_text.append(list(map(lambda mean_std_pair: "{0}±{1}".format(mean_std_pair[0], mean_std_pair[1]), stat)))


    plt.figure("Statistics Table")
    plt.table(colLabels=[NUM_FUNC_EVALS_STRING, TIME_ELAPSED_STRING, RELATIVE_DISTANCE_STRING],
              rowLabels=list(map(lambda func:func.func_name, funcs_to_test)),
              cellText=cell_text, loc='top')

    plt.show()

