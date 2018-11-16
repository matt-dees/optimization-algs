import qpsolvers
from numpy import *
import numpy
import stats_generator
import time
import scipy.optimize

M = array([[  80.69613754,  197.54341511,  186.43701167,   98.54964723,
         164.37171838,    9.50579752,   68.145483  ,  154.00383058,
         169.48552714,  172.72380941],
       [  34.70433164,   16.60245971,   76.15190389,   18.15800297,
         121.98984701,  194.74941502,   78.48914043,   24.55111931,
           5.83971525,   21.34164605],
       [  90.45237248,  109.9525822 ,  110.0320875 ,  102.38458083,
         172.42295615,  161.35723648,  160.54410401,   76.91975236,
          25.70611638,  115.44204323],
       [ 158.14914377,  140.10941826,    6.97604024,   43.8955222 ,
         145.93047681,  169.67423241,   54.50121987,   99.39273651,
         168.20215549,  146.73845218],
       [  90.21767006,  142.06945479,  198.43810048,  177.15544106,
         167.29301826,   72.94240731,   53.68668631,  126.99782335,
          28.50310634,   71.66276249],
       [  14.58114404,   46.01701953,  117.31195745,  105.13531344,
           3.60907301,   28.51901317,  189.24300084,  129.21358421,
          45.8422409 ,  178.48013532],
       [   1.73526545,  138.68711512,  144.87741972,  104.96990744,
         197.14148281,   92.1795195 ,  179.73987467,  100.00951112,
          54.96659061,  187.72828915],
       [  75.26125141,  119.98781851,  171.30441459,   36.78236051,
         140.79091813,  171.33580861,   48.52333065,  180.75702528,
          80.90227772,  102.83179743],
       [  47.2959556 ,  164.9578006 ,  123.84237216,  124.33504258,
          84.47609066,   37.04529343,    6.96773132,  180.37474901,
          10.7212393 ,   17.40343904],
       [  87.77507908,  192.56951201,  186.84885636,  167.97170772,
          67.2229434 ,  114.09052636,   21.56437041,    6.86327417,
         131.5040131 ,  137.8434112 ]])

P = dot(M.T, M)
q = dot(array([ 104.96545721,   89.52108804,  131.41880265,   48.0142488 ,
        117.87025338,   71.49959947,  187.47670014,   95.15310958,
        160.26497417,   73.06495801]), M).reshape((10,))

G = array([[ 15.80329383,  37.23892782,  47.7786633 ,   9.19228381,
         40.57733093,   9.90679605,  28.14676156,  39.32781356,
         37.12118315,  15.87228955],
       [ 43.27668213,  26.44885916,  41.42125893,  35.78354661,
         29.94610786,   2.85057763,  30.25055607,  33.45756588,
         46.38999009,  45.60324026],
       [ 39.44298274,  45.06131374,  35.03604878,  42.17292742,
         25.63598569,  24.64210732,   3.70977457,  20.5902921 ,
          1.27951776,  28.67218038],
       [ 45.65260265,  29.45461644,  45.97589935,  15.38474593,
         16.37986728,  36.72102556,  45.93156277,  37.22686868,
         40.22524935,   4.59660548],
       [  4.04383479,  44.73317497,  26.82846191,  15.3392613 ,
         30.79706094,  26.13894597,  13.59064127,  49.76311441,
         25.02106243,   6.30114704],
       [ 41.98628128,  25.9705214 ,  10.28335199,   2.77472287,
          5.7108503 ,  23.45671499,   2.71313764,   3.7822363 ,
         18.61527093,   4.42257031],
       [ 27.76025925,  40.3914235 ,  43.67169401,  34.49917188,
          2.00756052,  10.30928905,  38.53562056,   9.20011528,
         11.82462632,  46.58849645],
       [ 25.61764567,  36.31465585,  45.66559564,  48.96504435,
         25.94750581,  36.26049   ,  24.46752968,  39.84317833,
         16.31109554,  22.66888219],
       [ 34.61476803,  17.50168948,  32.58135685,   0.10132606,
         28.63538219,  42.87073243,  31.79667229,  13.60056144,
         31.79662649,   8.5557422 ],
       [ 44.59563999,   3.25900081,  36.96505858,  19.68861835,
          8.53496148,  21.61004563,  35.50452822,  49.29895152,
          3.30418814,  29.34439322]])
h = array([  77.92369082,    1.57602341,   24.33257481,  121.40012023,
        114.21005826,  185.20003602,   13.87404591,   82.31918607,
        120.27771735,  103.28880513])


class QuadConstrained:

    @staticmethod
    def cost(x):
        return .5 * x.T.dot(P).dot(x) + q.T.dot(x)


class QuadUnconstrained:

    @staticmethod
    def penalty(x):
        penalty = 0
        for i in range(len(x)):
            if x.dot(G[i]) > h[i]:
                penalty += 1e5 * (x.dot(G[i]) - h[i]) ** 2
        return penalty

    @staticmethod
    def cost(x):
        return .5 * x.T.dot(P).dot(x) + q.T.dot(x) + QuadUnconstrained.penalty(x)


def test_constrained_quad_solver(method, iters):
    results = []
    for i in range(iters):
        start = time.time()
        res = qpsolvers.solve_qp(P, q, G, h, solver=method)
        end = time.time()
        results.append((QuadConstrained.cost(res), (end-start) * 1000))

    func_vals = list(map(lambda x: x[0], results))
    time_vals = list(map(lambda x: x[1], results))

    return ((stats_generator.StatsGenerator.unbiased_expected_value(func_vals),
             stats_generator.StatsGenerator.std_dev(func_vals)),
            (stats_generator.StatsGenerator.unbiased_expected_value(time_vals),
             stats_generator.StatsGenerator.std_dev((time_vals))))


def test_unconstrained(method, iters):
    print("TESTING UNCONSTRAINED METHOD:", method)
    test_results = []

    for i in range(iters):
        x0 = numpy.random.rand(10) * 100
        start = time.time()
        res = scipy.optimize.minimize(QuadUnconstrained.cost, x0=x0, method=method)
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
    print("Avg. time value (ms): ", stats_generator.StatsGenerator.stats_to_string(stats[1]))
    print("\n")


if __name__ == "__main__":
    # stats = test_constrained_quad_solver("quadprog", 1000)
    # print_stats("QPSOLVERS QUADPROG", stats)
    #
    # stats = test_constrained_quad_solver("cvxopt", 1000)
    # print_stats("QPSOLVERS CVXOPT", stats)

    stats = test_unconstrained("cg", 100)

    print_stats("SCIPY UNCONSTRAINED", stats)