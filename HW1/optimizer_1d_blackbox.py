import collections
import time
import math
from function_test_harness import TestHarness


class Optimizer1D:
    """
    Utility class used to optimize unconstrained 1D functions.
    """

    TAU = (math.sqrt(5) - 1) / 2
    TAU_INVERSE = 1 / TAU
    EPSILON_MACHINE = 1.11e-16
    EPSILON_ABSOLUTE = 1.11e-14
    EPSILON_RELATIVE = math.sqrt(EPSILON_MACHINE)

    # Used to represent three point interval
    OptOutput = collections.namedtuple("OptOutput", "x f_x num_function_calls")

    def __init__(self, func):
        self._num_function_calls = 0
        self._optimizing_function = func

    class TimeExpired(Exception):
        pass

    @classmethod
    def _should_stop(cls, interval_delta, middle_point, epsilon_rel, epsilon_abs):
        """
        Evaluate stopping criteria condition.
        :param interval_delta: Distance between first and last interval points
        :param middle_point: Point selected between two interval points (middle of 3 point pattern)
        :param epsilon_rel: Relative epsilon value. Calculated using machine espilson
        :param epsilon_abs: Absolute epsilon value. Calculated using the relative epsilon
        :return: True if stopping criteria met for the given parameters.
        """
        return abs(interval_delta) <= epsilon_rel * abs(middle_point) + epsilon_abs

    @classmethod
    def _descent_function(cls, f_1, f_2, f_3):
        """
        Calculates the descent function (essentially an average of outputs).
        :param f_1: Output for point 1
        :param f_2: Output for point 2
        :param f_3: Output for point 3
        :return: Descent function calculated (float)
        """
        return (f_1 + f_2 + f_3) / 3

    @classmethod
    def _timer_expired(cls, time_elapsed, expiration_time):
        """
        Returns true if the timer has expired.
        :param time_elapsed: Amount of time that has passed
        :param expiration_time: Amount of time allotted.
        :return:
        """
        return time_elapsed >= expiration_time

    def _evaluate_function(self, func, *args, **kwargs):
        """
        Calls function and increase increment.
        :param func: Function to eval.
        :param args: Args to pass to function.
        :param kwargs: Keyword arguments to pass to function.
        :return: Function evaluation.
        """
        self._num_function_calls += 1
        return func(*args, **kwargs)

    def golden_section(self, func, start, step_size, time_constraint=5):
        """
        Finds the optimum (x, y) for function <func> given the following parameters. Uses an expansion factor
        of <GOLDEN_RATIO> for the integrated Bracketing Algorithm.
        :param func: function that will be evaluated
        :param start: starting point
        :param step_size: increment
        :param time_constraint: amount of time for optimization to complete
        :return: OptOutput
        """
        # Reset function call counter
        self._num_function_calls = 0

        # Save start time for run time constraint
        time_start = time.time()

        # Calculate first point in the interval
        x_1 = start
        f_1 = self._evaluate_function(func, x_1)

        # Save step size as s
        s = step_size

        # Determine second point and calculate function output
        x_2 = x_1 + s
        f_2 = self._evaluate_function(func, x_2)

        # Set old function descent value to None (populated on second iteration)
        f_descent_old = None

        # If F_2 is larger than F_1 we are going the wrong way. Turn around.
        if f_2 > f_1:
            f_1, f_2 = f_2, f_1
            x_1, x_2 = x_2, x_1
            s = -s

        while True:
            # Check to see if timer expired. Don't want to run forever.
            if Optimizer1D._timer_expired(time.time() - time_start, time_constraint):
                raise Optimizer1D.TimeExpired

            # Divide step size by tau to determine offset of last interval point. Want to increase step_size
            # by a greater value each time to find 3 point pattern more quickly.
            s /= Optimizer1D.TAU

            # Calculate x and y value for last interval point
            x_4 = x_2 + s
            f_4 = self._evaluate_function(func, x_4)

            # If the output of the last interval point is greater than our "middle" point, we have found a
            # 3 point pattern.
            if f_4 > f_2:
                break

            # Continue moving in the positive direction until we find a pattern.
            x_1, x_2 = x_2, x_4
            f_1, f_2 = f_2, f_4

        while True:

            # Check to see if timer expired. Don't want to run forever.
            if Optimizer1D._timer_expired(time.time() - time_start, time_constraint):
                raise Optimizer1D.TimeExpired

            # Calculate where the next point should be. Want to maintain a ratio of 1 - tau : tau.
            x_3 = Optimizer1D.TAU * x_4 + (1 - Optimizer1D.TAU) * x_1
            f_3 = self._evaluate_function(func, x_3)

            # Calculate current function descent value that will be used in stopping criteria.
            f_descent_curr = Optimizer1D._descent_function(f_1, f_2, f_3)

            # Determine if the algorithm should stop.
            if f_descent_old and (Optimizer1D._should_stop(x_1 - x_3, x_2, Optimizer1D.EPSILON_RELATIVE,
                                                           Optimizer1D.EPSILON_ABSOLUTE) or
                Optimizer1D._should_stop(f_descent_curr - f_descent_old, f_2, Optimizer1D.EPSILON_RELATIVE,
                                         Optimizer1D.EPSILON_ABSOLUTE)):
                return Optimizer1D.OptOutput(x_2, f_2, self._num_function_calls)

            # If f_2 < f_3 go to the left, otherwise go to the right.
            if f_2 < f_3:
                x_4, x_1 = x_1, x_3
            else:
                x_1, x_2, f_2 = x_2, x_3, f_3

            # Save old function descent output.
            f_descent_old = f_descent_curr

    def optimize(self, func, *args):
        """
        This function should be used to find the optimal values for function <func>
        :param func: Function to find optimal values for.
        :param args: Arguments to pass to the function.
        :return: OptOutput
        """
        return self._optimizing_function(self, func, *args)


if __name__ == "__main__":
    
    test_harness = TestHarness()
    test_harness.load_optimizer(Optimizer1D(Optimizer1D.golden_section))
    test_harness.load_test_function(lambda x: (x - 2) ** 2)
    report_list = test_harness.test_optimizer([-1000000, 2, 3], [1, 1, 10])
    print(report_list)