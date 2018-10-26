from stats_generator import StatsGenerator
from function_test_harness import TestHarness
from nelder_mead import nelder_mead
from SA import simulated_annealing
from test_functions import KnapsackProblem, GraphColoring
import random
from csv_formatter import CSVFormatter


def problem_1_a():

    csvf = CSVFormatter()
    csvf.add_row(["Problem Size", "Time (ms)", "Profit"])
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

    weights = [7, 0, 30, 22, 80, 94, 11, 81, 70,
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
          79, 20, 65, 52, 13]

    num_runs = 5
    knapsack_problem_size_range = (5, 125, 5)
    max_weight_modifier = 37

    harness_param_t0 = [25000] * num_runs
    harness_param_t_final = [0.1] * num_runs

    for problem_size in range(*knapsack_problem_size_range):
        # create knapsack problem for test
        kp = KnapsackProblem(max_weight=max_weight_modifier * problem_size, values=values[:problem_size],
                             weights=weights[:problem_size])

        harness_param_x0 = [[round(random.uniform(0, 1)) for _ in range(problem_size)] for _ in range(num_runs)]
        allowed = [[0, 1] for _ in range(problem_size)]
        harness_param_allowed = [allowed] * num_runs

        run_report_list = TestHarness.test_optimizer(simulated_annealing, kp.get_profit, harness_param_x0, harness_param_allowed, harness_param_t0,
                                   harness_param_t_final)

        time_stats = StatsGenerator.generate_stats(list(map(lambda x: x[0], run_report_list)))
        profit_stats = StatsGenerator.generate_stats(list(map(lambda x: x[2], run_report_list)))

        table_entry = [problem_size, StatsGenerator.stats_to_string(time_stats), StatsGenerator.stats_to_string(profit_stats)]
        csvf.add_row(table_entry)
        print("Finished problem size:", problem_size)
        print(table_entry)

    csvf.export_table("sa_knapsack_test.csv")


def problem_1_b():
    csvf = CSVFormatter()
    csvf.add_row(["Time (ms)"])
    color_set = range(3)
    problem_size = 10
    num_runs = 100

    cg = GraphColoring(color_set)

    harness_param_x0 = [[random.choice(color_set) for _ in range(problem_size)] for _ in range(num_runs)]
    harness_allowed = [[color_set for _ in range(problem_size)]] * num_runs
    harness_t0 = [25000] * num_runs
    harness_t_final = [0.1] * num_runs
    run_report_list = TestHarness.test_optimizer(simulated_annealing, cg.solve, harness_param_x0, harness_allowed, harness_t0, harness_t_final)

    print("Finished coloring graph runs")
    time_stats = StatsGenerator.stats_to_string(StatsGenerator.generate_stats(list(map(lambda x:x[0], run_report_list))))
    completion_stats = StatsGenerator.stats_to_string(StatsGenerator.generate_stats(list(map(lambda x:x[2], run_report_list))))
    print(time_stats, completion_stats)
    csvf.add_row([time_stats, completion_stats])

    csvf.export_table("sa_graph_coloring")


def problem_2():

    nm_csvf = CSVFormatter()
    sa_csvf = CSVFormatter()

    nm_csvf.add_row(["Time (ms)", "Minimum"])
    sa_csvf.add_row(["Time (ms)", "Minimum"])

    nm_test_function = lambda x: x[0] ** 2 + 2 * x[1] ** 2 + 2 * x[0] * x[1]

    num_runs = 1000

    harness_sa_allowed = [[[], []]] * num_runs
    harness_sa_t0 = [25000] * num_runs
    harness_sa_t_final = [0.1] * num_runs

    harness_x0 = [(random.uniform(-100, 100), random.uniform(-100, 100)) for _ in range(num_runs)]

    nm_run_report = TestHarness.test_optimizer(nelder_mead, nm_test_function, harness_x0)
    sa_run_report = TestHarness.test_optimizer(simulated_annealing, nm_test_function, harness_x0, harness_sa_allowed, harness_sa_t0, harness_sa_t_final)

    nm_time_stats = StatsGenerator.generate_stats(list(map(lambda x:x[0], nm_run_report)))
    nm_min_stats = StatsGenerator.generate_stats(list(map(lambda x:x[2], nm_run_report)))

    sa_time_stats = StatsGenerator.generate_stats(list(map(lambda x:x[0], sa_run_report)))
    sa_min_stats = StatsGenerator.generate_stats(list(map(lambda x:x[2], sa_run_report)))

    nm_csvf.add_row([StatsGenerator.stats_to_string(nm_time_stats), StatsGenerator.stats_to_string(nm_min_stats)])
    sa_csvf.add_row([StatsGenerator.stats_to_string(sa_time_stats), StatsGenerator.stats_to_string(sa_min_stats)])

    nm_csvf.export_table("tables/nelder_mead_stats.csv")
    sa_csvf.export_table("tables/continuous_simulated_annealing_stats.csv")


if __name__ == "__main__":
    #problem_1_a()
    #problem_1_b()
    problem_2()