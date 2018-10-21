from conjugate_gradient import conjugate_gradients, alternate_method
from stats_generator import StatsGenerator
from function_test_harness import TestHarness
import random
from csv_formatter import CSVFormatter

NUM_RUNS = 100
NUM_VARIATIONS = 50
CELL_FORMAT_STR = "{0:.2e}±{1:.2e}"
CG_HEADER = ["restart_frequency", "epsilon", "x_error(%)", "f_error(%)", "iterations"]
STEEP_HEADER = CG_HEADER[1:-1]

def run_cg_func_1():
    csv_table = CSVFormatter()

    ## FIRST FUNCTION
    ## Rosenblock
    test_func = lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2 + 1
    fprime = lambda x: (2 * (200 * x[0] ** 3 - 200 * x[0] * x[1] + x[0] - 1),
                        200 * (x[1] - x[0] ** 2))

    # Vary GC restart frequency (2->50 drawn from uniform distribution

    # Func 1 conj variation
    epsilon_abs_param_dist = [1.14e-14] * NUM_RUNS
    x0_param = [(random.gauss(-10, 10), random.gauss(-10, 10)) for _ in range(NUM_RUNS)]

    csv_table.add_row(CG_HEADER)
    for i in range(NUM_VARIATIONS):
        gc_iterations_param_dist = [random.randint(2, NUM_VARIATIONS)] * NUM_RUNS
        run_report_list = TestHarness.test_optimizer(conjugate_gradients, test_func, x0_param,
                                                     [fprime for _ in range(NUM_RUNS)],
                                                     gc_iterations_param_dist, epsilon_abs_param_dist)
        stats = StatsGenerator.generate_stats(run_report_list, (1, 1), 1)
        print(stats)
        csv_table.add_row(
            [gc_iterations_param_dist[i], epsilon_abs_param_dist[i], CELL_FORMAT_STR.format(stats[0][0], stats[0][1]),
             CELL_FORMAT_STR.format(stats[1][0], stats[1][1]), CELL_FORMAT_STR.format(stats[2][0], stats[2][1])])

    csv_table.export_table("function1_gc_freq_variation.csv")
    csv_table.clear_table()
    print("Finished function 1 GC restart variation")

    csv_table.add_row(CG_HEADER)
    # Func 1 epsilon variation
    eps = 1e-14
    for i in range(10):
        gc_iterations_param_dist = [10] * NUM_RUNS
        epsilon_abs_param_dist = [eps] * NUM_RUNS
        run_report_list = TestHarness.test_optimizer(conjugate_gradients, test_func, x0_param,
                                                     [fprime for _ in range(NUM_RUNS)],
                                                     gc_iterations_param_dist, epsilon_abs_param_dist)
        stats = StatsGenerator.generate_stats(run_report_list, (1, 1), 1)
        print(stats)
        csv_table.add_row(
            [gc_iterations_param_dist[i], epsilon_abs_param_dist[i], CELL_FORMAT_STR.format(stats[0][0], stats[0][1]),
             CELL_FORMAT_STR.format(stats[1][0], stats[1][1]), CELL_FORMAT_STR.format(stats[2][0], stats[2][1])])
        eps *= 10

    csv_table.export_table("function1_epsilon_variation.csv")
    print("Finished function 1 epsilon variation")
    csv_table.clear_table()

def run_cg_func_2():
    csv_table = CSVFormatter()

    test_func = lambda x: x[0] * x[1] + sum([(ele - x[ele - 1]) ** 2 for ele in range(1, 11)])
    fprime = lambda x: [x[1] + -2 * (1 - x[0]),
                        x[0] + -2 * (2 - x[1])] + \
                        [-2*(ele - x[ele - 1]) for ele in range(3, 11)]

    x0_param = [
        [random.gauss(-10, 10) for _ in range(10)] for _ in range(NUM_RUNS)
    ]
    epsilon_abs_param_dist = [1.14e-14] * NUM_RUNS

    csv_table.add_row(CG_HEADER)
    for i in range(NUM_VARIATIONS):
        gc_iterations_param_dist = [random.randint(2, NUM_VARIATIONS)] * NUM_RUNS
        run_report_list = TestHarness.test_optimizer(conjugate_gradients, test_func, x0_param,
                                                     [fprime for _ in range(NUM_RUNS)],
                                                     gc_iterations_param_dist, epsilon_abs_param_dist)
        stats = StatsGenerator.generate_stats(run_report_list,  (0, 2, 3, 4, 5, 6, 7, 8, 9, 10), 1)
        print(stats)
        csv_table.add_row(
            [gc_iterations_param_dist[i], epsilon_abs_param_dist[i], CELL_FORMAT_STR.format(stats[0][0], stats[0][1]),
             CELL_FORMAT_STR.format(stats[1][0], stats[1][1]), CELL_FORMAT_STR.format(stats[2][0], stats[2][1])])

    csv_table.export_table("function2_gc_freq_variation.csv")
    csv_table.clear_table()
    print("Finished function 2 GC restart variation")

    csv_table.add_row(CG_HEADER)
    # Func 1 epsilon variation
    eps = 1e-14
    for i in range(10):
        gc_iterations_param_dist = [10] * NUM_RUNS
        epsilon_abs_param_dist = [eps] * NUM_RUNS
        run_report_list = TestHarness.test_optimizer(conjugate_gradients, test_func, x0_param,
                                                     [fprime for _ in range(NUM_RUNS)],
                                                     gc_iterations_param_dist, epsilon_abs_param_dist)
        stats = StatsGenerator.generate_stats(run_report_list, (0, 2, 3, 4, 5, 6, 7, 8, 9, 10), 1)
        print(stats)
        csv_table.add_row(
            [gc_iterations_param_dist[i], epsilon_abs_param_dist[i], CELL_FORMAT_STR.format(stats[0][0], stats[0][1]),
             CELL_FORMAT_STR.format(stats[1][0], stats[1][1]), CELL_FORMAT_STR.format(stats[2][0], stats[2][1])])
        eps *= 10

    csv_table.export_table("function2_epsilon_variation.csv")
    print("Finished function 2 epsilon variation")
    csv_table.clear_table()

def run_steepest_func_1():
    csv_table = CSVFormatter()

    ## FIRST FUNCTION
    ## Rosenblock
    test_func = lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2 + 1
    fprime = lambda x: (2 * (200 * x[0] ** 3 - 200 * x[0] * x[1] + x[0] - 1),
                        200 * (x[1] - x[0] ** 2))

    csv_table.add_row(STEEP_HEADER)
    # Func 1 epsilon variation
    x0_param = [(random.gauss(-10, 10), random.gauss(-10, 10)) for _ in range(NUM_RUNS)]


    eps = 1e-14
    for i in range(10):
        gc_iterations_param_dist = [10] * NUM_RUNS
        epsilon_abs_param_dist = [eps] * NUM_RUNS
        run_report_list = TestHarness.test_optimizer(alternate_method, test_func, x0_param,
                                                     [fprime for _ in range(NUM_RUNS)], epsilon_abs_param_dist)
        stats = StatsGenerator.generate_stats(run_report_list, (1,1), 1)
        print(stats)
        csv_table.add_row(
            [epsilon_abs_param_dist[i], CELL_FORMAT_STR.format(stats[0][0], stats[0][1]),
             CELL_FORMAT_STR.format(stats[1][0], stats[1][1])])
        eps *= 10

    csv_table.export_table("steepest_func1.csv")
    print("Finished steepest decent function 1")
    csv_table.clear_table()

def run_steepest_func_2():
    csv_table = CSVFormatter()

    # x_0 * x_1 + (1-x_0)^2 + (2-x_1)^2+ (3-x_2)^2+ (4-x_3)^2+ (5-x_4)^2+ (6-x_5)^2+ (7-x_6)^2+ (8-x_7)^2+ (9-x_8)^2+ (10-x_9)^2
    ## 10 Dimension
    test_func = lambda x: x[0] * x[1] + sum([(ele - x[ele - 1]) ** 2 for ele in range(1, 11)])
    fprime = lambda x: [x[1] + -2 * (1 - x[0]),
                        x[0] + -2 * (2 - x[1])] + \
                        [-2*(ele - x[ele - 1]) for ele in range(3, 11)]

    csv_table.add_row(STEEP_HEADER)

    x0_param = [
        [random.gauss(-10, 10) for _ in range(10)] for _ in range(NUM_RUNS)
    ]

    eps = 1e-14
    for i in range(10):
        gc_iterations_param_dist = [10] * NUM_RUNS
        epsilon_abs_param_dist = [eps] * NUM_RUNS
        run_report_list = TestHarness.test_optimizer(alternate_method, test_func, x0_param,
                                                     [fprime for _ in range(NUM_RUNS)], epsilon_abs_param_dist)
        stats = StatsGenerator.generate_stats(run_report_list, (1, 1), 1)
        print(stats)
        csv_table.add_row(
            [epsilon_abs_param_dist[i], CELL_FORMAT_STR.format(stats[0][0], stats[0][1]),
             CELL_FORMAT_STR.format(stats[1][0], stats[1][1])])
        eps *= 10

    csv_table.export_table("steepest_func2.csv")
    print("Finished steepest decent function 2")
    csv_table.clear_table()
if __name__ == "__main__":

    run_cg_func_1()
    run_cg_func_2()
    run_steepest_func_1()
    run_steepest_func_2()