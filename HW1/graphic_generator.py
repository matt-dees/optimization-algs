from texttable import Texttable
import matplotlib.pyplot as plt
from stats_generator import StatsGenerator
from random import randint


class GraphicGenerator:
    NUM_FUNC_EVALS_STRING = "Number of function evaluations"
    TIME_ELAPSED_STRING = "Time elapsed (ms)"
    RELATIVE_DISTANCE_STRING = "Relative Distance from Optimum (%)"
    COLUMN_HEADER_PREFIX = 'Avg. '
    CELL_FORMAT_STR = "{0:.5e} ± {1:.5e}"

    @classmethod
    def generate_stats_table(cls, functions_under_test, statistics_from_run):
        table = Texttable()

        rows = [['Function', cls.COLUMN_HEADER_PREFIX + cls.NUM_FUNC_EVALS_STRING, cls.COLUMN_HEADER_PREFIX + cls.TIME_ELAPSED_STRING,
                 cls.COLUMN_HEADER_PREFIX + cls.RELATIVE_DISTANCE_STRING]]
        for i, func in enumerate(functions_under_test):
            row = [func.func_name,
                   cls.CELL_FORMAT_STR.format(statistics_from_run[i].function_evals[0], statistics_from_run[i].function_evals[1]),
                   cls.CELL_FORMAT_STR.format(statistics_from_run[i].time_elapsed[0], statistics_from_run[i].time_elapsed[1]),
                   cls.CELL_FORMAT_STR.format(statistics_from_run[i].relative_distance[0], statistics_from_run[i].relative_distance[1])]
            rows.append(row)

        table.add_rows(rows)
        print(table.draw())

    @classmethod
    def construct_box_plot(cls, graph_name, function_name, data):
        plt.figure(function_name + " " + graph_name)
        plt.title(function_name)
        plt.ylabel(graph_name)
        plt.xlabel("")
        plt.boxplot(data)
        plt.savefig("boxplots/" + function_name + "_" + graph_name + ".png")

    @classmethod
    def generate_box_plots(cls, function_under_test, run_report_list):
        num_function_evals_data = [report.num_function_calls for report in run_report_list]
        time_elapsed_ms_data = [report.time for report in run_report_list]
        optimum_distance_data = [StatsGenerator.relative_distance(function_under_test.optimum, report.x_min) for report in run_report_list]

        cls.construct_box_plot(cls.NUM_FUNC_EVALS_STRING, function_under_test.func_name, num_function_evals_data)
        cls.construct_box_plot(cls.TIME_ELAPSED_STRING, function_under_test.func_name, time_elapsed_ms_data)
        cls.construct_box_plot(cls.RELATIVE_DISTANCE_STRING, function_under_test.func_name, optimum_distance_data)
