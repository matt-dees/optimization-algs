import scipy.optimize
import numpy
import time
import stats_generator


FactoryCostCoefficients = [.8, 1, .95, .72, .9, .87, .72, .82, .93, .67]
FactoryConstraints = [
        1000, 1000, 3000, 1500, 1600, 750, 2000, 1100, 500, 350, 1500
    ]


class ConstrainedFactoryTaxProblem:
    c = numpy.asarray(FactoryCostCoefficients) * -1
    A_u = numpy.eye(10)
    A_u = numpy.append(A_u, [[1, 1] + [0] * 8], axis=0)
    b_u = FactoryConstraints

    A_eq = numpy.zeros((10, 10))
    A_eq[0] = [0] * 5 + [1, 1, 1, 0, 0]

    b_eq = [1800] + [0] * 9


class UnconstrainedFactoryProblem:
    PUNISHMENT = 1e6

    @staticmethod
    def penalty(x):
        total_penalty = 0
        ## Greater than zero constraint
        for i in range(len(x)):
            total_penalty += UnconstrainedFactoryProblem.PUNISHMENT * min(x[i], 0) ** 2
        ## LE constraints
        # x < b
        for i in range(len(x)):
            total_penalty += UnconstrainedFactoryProblem.PUNISHMENT * max(0, x[i] - FactoryConstraints[i]) ** 2
        # x_0 + x_ 1 <= 1500
        total_penalty += UnconstrainedFactoryProblem.PUNISHMENT * max(0, x[0] + x[1] - 1500) ** 2

        ## EQ constraints
        fifth_factory_sum = x[5] + x[6] + x[7]
        total_penalty += UnconstrainedFactoryProblem.PUNISHMENT * (fifth_factory_sum - 1800) ** 2
        return total_penalty

    @staticmethod
    def cost(x):
        return sum(numpy.asarray(FactoryCostCoefficients) * x * -1) + UnconstrainedFactoryProblem.penalty(x)


def test_scipy_method(method, iters):
    test_results = []
    for i in range(iters):
        start = time.time()
        res = scipy.optimize.linprog(c=ConstrainedFactoryTaxProblem.c, A_eq=ConstrainedFactoryTaxProblem.A_eq, b_eq=ConstrainedFactoryTaxProblem.b_eq,
                                     A_ub=ConstrainedFactoryTaxProblem.A_u, b_ub=ConstrainedFactoryTaxProblem.b_u, method=method)
        end = time.time()
        test_results.append((res.fun, (end-start) * 1000))

    func_vals = list(map(lambda x: x[0], test_results))
    time_vals = list(map(lambda x: x[1], test_results))

    return ((stats_generator.StatsGenerator.unbiased_expected_value(func_vals), stats_generator.StatsGenerator.std_dev(func_vals)),
            (stats_generator.StatsGenerator.unbiased_expected_value(time_vals), stats_generator.StatsGenerator.std_dev((time_vals))))


def test_unconstrained_lp(method, iters):
    print("TESTING UNCONSTRAINED METHOD:", method)
    test_results = []

    for i in range(iters):
        x0 = numpy.random.rand(10) * 3000
        start = time.time()
        res = scipy.optimize.minimize(UnconstrainedFactoryProblem.cost, x0=x0, method=method)
        if i == 0:
            print(res)
        end = time.time()
        test_results.append((res.fun, (end-start) * 1000))

    func_vals = list(map(lambda x: x[0], test_results))
    time_vals = list(map(lambda x: x[1], test_results))
    return ((stats_generator.StatsGenerator.unbiased_expected_value(func_vals), stats_generator.StatsGenerator.std_dev(func_vals)),
            (stats_generator.StatsGenerator.unbiased_expected_value(time_vals), stats_generator.StatsGenerator.std_dev((time_vals))))


def print_stats(header, stats):
    print(header)
    print("Avg. function value: ", stats_generator.StatsGenerator.stats_to_string(stats[0]))
    print("Avg. time value: ", stats_generator.StatsGenerator.stats_to_string(stats[1]))
    print("\n")


if __name__ == "__main__":
    stats = test_scipy_method("simplex", 1000)
    print_stats("SCIPY SIMPLEX METHOD", stats)

    stats = test_scipy_method("interior-point", 1000)
    print_stats("SCIPY INTERIOR-POINT METHOD", stats)

    stats = test_unconstrained_lp("cg", 20)
    print_stats("SCIPY UNCONSTRAINED", stats)
