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
            return cls.euclidean_dis1tance(x_optimum, x_code) / math.sqrt(sum([x**2 for x in x_optimum]))
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
    def generate_stats(cls, data):
        return cls.unbiased_expected_value(data), cls.std_dev(data)

    @classmethod
    def stats_to_string(cls, stats):
        stats_str = "{0:.2e}±{1:.2e}"
        return stats_str.format(*stats)