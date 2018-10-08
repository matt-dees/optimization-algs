import matplotlib.pyplot as plt
from collections import namedtuple

Stats = namedtuple("Stats", "function_evals time_elapsed relative_distance")

class StatsGenerator:



    @classmethod
    def relative_distance(cls, x_optimum, x_code):
        return 100 * (abs(x_code - x_optimum) / (abs(x_optimum)))

    @classmethod
    def construct_box_plot(cls, graph_name, function_name, data):
        plt.figure(function_name + " " + graph_name)
        plt.title(function_name)
        plt.ylabel(graph_name)
        plt.boxplot(data)

    @classmethod
    def unbiased_expected_value(cls, data):
        if len(data) == 0:
            return 0
        return sum(data) / len(data)

    @classmethod
    def std_dev(cls, data):
        if len(data) <= 1:
            return 0
        avg_val = cls.unbiased_expected_value(data)
        variance_list = map(lambda x : (x - avg_val) ** 2, data)
        bessel_correction = 1 / (len(data) - 1)
        return bessel_correction * sum(variance_list)

    @classmethod
    def generate_stats(cls, run_report_list, optimum, function_name):
        num_function_evals_data = [report.num_function_calls for report in run_report_list]
        time_elapsed_ms_data = [report.time for report in run_report_list]
        optimum_distance_data = [StatsGenerator.relative_distance(optimum, report.x_min) for report in run_report_list]

        return Stats(function_evals=(cls.unbiased_expected_value(num_function_evals_data),
                                     cls.std_dev(num_function_evals_data)),
                     time_elapsed=(cls.unbiased_expected_value(time_elapsed_ms_data),
                                   cls.std_dev(time_elapsed_ms_data)),
                     relative_distance=(cls.unbiased_expected_value(optimum_distance_data),
                                        cls.std_dev(optimum_distance_data)))
