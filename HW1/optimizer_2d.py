from optimizer_1d import Optimizer1D
import math
from utilies import FuncCallCounter


def optimizer2D(func, initial_point, initial_step_size):
    """
    Optimizer 2D function that uses the prototype provided in part 1 of the homework.
    :param func: Function in the form of f(x,y)
    :param initial_point: (x_0, y_0) representing the starting point for the optimizer
    :param initial_step_size: (x_delta, y_delta) representing the steps for each point
    :return: point in the when the optimum value can be computed (x, y)
    """
    return Optimizer2D.coordinate_descent(func, initial_point, initial_step_size).minimizing_input


class Optimizer2D:

    EPSILON_MACHINE = 1.11e-16
    EPSILON_ABSOLUTE = 1.11e-14
    EPSILON_RELATIVE = math.sqrt(EPSILON_MACHINE)

    @classmethod
    def should_stop(cls, x_delta, y_delta, f_delta):
        """
        Stopping criteria for 2D optimizer. Check delta for input and output and compare
        to relative machine epsilon.

        :param x_delta:
        :param y_delta:
        :param f_delta:
        :return:
        """
        input_should_stop = abs(x_delta) <= abs(x_delta)/2 * cls.EPSILON_RELATIVE + cls.EPSILON_ABSOLUTE and \
                            abs(y_delta) <= abs(y_delta)/2 * cls.EPSILON_RELATIVE + cls.EPSILON_ABSOLUTE
        output_should_stop = abs(f_delta) <= abs(f_delta)/2 * cls.EPSILON_RELATIVE + cls.EPSILON_ABSOLUTE
        return input_should_stop or output_should_stop

    @classmethod
    def coordinate_descent(cls, func, initial_point, initial_step_size):
        """
        Coordinate descent algorithm using the 1D optimizer pulled in through the
        optimizer_1d.py file.

        :param x0_start: Starting point for x0
        :param initial_step_size: Initial increment value.
        :return: RunReport
        """
        func = FuncCallCounter(func)

        current_x = initial_point[0]
        current_y = initial_point[1]

        f_old = None
        x_old = None
        y_old = None

        while True:
            opt_report = Optimizer1D.golden_section(lambda x: func(x, current_y), current_x, initial_step_size[0])
            current_x = opt_report.minimizing_input

            opt_report = Optimizer1D.golden_section(lambda y: func(current_x, y), current_y, initial_step_size[1])
            current_y = opt_report.minimizing_input

            f_curr = func(current_x, current_y)

            is_gt_second_iteration = f_old is not None and x_old is not None and y_old is not None
            if is_gt_second_iteration and cls.should_stop(current_x - x_old, current_y - y_old, f_curr - f_old):
                return Optimizer1D.OptOutput(minimizing_input=(current_x, current_y), minimized_output=f_curr, num_function_calls=func.get_num_calls())

            f_old = f_curr
            x_old = current_x
            y_old = current_y


if __name__ == "__main__":
    func = lambda x, y : 5*math.pow(x - 2.5, 2) - 6*(x - 2.5)*(y - 20.5) + 5*math.pow(y - 20.5, 2)
    func2 = lambda x, y: abs(x + 2.0) + abs(y - 1.0)
    print(optimizer2D(func, (0, 0), (1,1)))
    print(optimizer2D(func2, (0,0), (1,1)))