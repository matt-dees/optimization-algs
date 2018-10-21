from collections import namedtuple, Iterable
import math

Stats = namedtuple("Stats", "input_avg_error output_avg_error num_gc_iterations")


class StatsGenerator:

    @classmethod
    def euclidean_distance(cls, p1, p2):
        return math.sqrt(sum([(p1_ele - p2_ele) ** 2 for (p1_ele, p2_ele) in zip(p1, p2)]))

    @classmethod
    def relative_distance(cls, x_optimum, x_code):
        if isinstance(x_optimum, Iterable) and isinstance(x_code, Iterable):
            return cls.euclidean_distance(x_optimum, x_code) / math.sqrt(sum([x**2 for x in x_optimum]))
        return 100 * (abs(x_code - x_optimum) / (abs(x_optimum)))

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
        variance_list = list(map(lambda x : (x - avg_val) ** 2, data))
        bessel_correction = 1 / (len(data) - 1)
        return math.sqrt(bessel_correction * sum(variance_list))

    @classmethod
    def generate_stats(cls, run_report_list, optimum_input, optimum_output):
        input_vector_data = [report[1] for report in run_report_list]
        output_vector_data = [report[2] for report in run_report_list]


        avg_input_error = [StatsGenerator.relative_distance(ele, optimum_input) for ele in input_vector_data]
        avg_output_error = [StatsGenerator.relative_distance(ele, optimum_output) for ele in output_vector_data]
        ret = [(cls.unbiased_expected_value(avg_input_error),
                                     cls.std_dev(avg_input_error)),
                     (cls.unbiased_expected_value(avg_output_error),
                                   cls.std_dev(avg_output_error))]

        if len(run_report_list[0]) > 3:
            gc_iters = [report[3] for report in run_report_list]
            ret.append((cls.unbiased_expected_value(gc_iters), cls.std_dev(gc_iters)))

        return ret

