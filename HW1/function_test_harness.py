import time
import collections

RunReport = collections.namedtuple("RunReport", "status time num_function_calls x_min y_min")


class TestHarness:

    ERROR = 1
    SUCCESS = 0

    def __init__(self):
        self._optimizer = None
        self._test_function = None

    class RunFailed(Exception):

        def __init__(self, optimizer, func, *args):
            self._args = args
            self._optimizer = optimizer
            self._func = func

        def __str__(self):
            return "Args: {0}, Optimizer: {1}, Function: {2}".format(self._args, self._optimizer, self._func)

    def load_optimizer(self, optimizer):
        self._optimizer = optimizer

    def load_test_function(self, test_function):
        self._test_function = test_function

    def test_optimizer(self, *args):
        arg_array_lens = [len(arg) for arg in args]
        if len(set(arg_array_lens)) != 1:
            print("Invalid usage. Input arguments must be arrays of equal lengths representing parameters for each run." \
                  "\nExample: func, [1, 2, 3], [.1 .01, .1] will run func() three times with the respective parameters" \
                  "during the runs.")
            return

        num_runs = len(args[0])
        run_report_list = []
        for run in range(num_runs):
            params = [arg[run] for arg in args]
            run_report_list.append(self.run_single_instance(*params))
        return run_report_list

    def run_single_instance(self, *args):
        start = time.time()
        try:
            opt_output = self._optimizer.optimize(self._test_function, *args)
        except Exception as e:
            print(e)
            return RunReport(1, 0, 0, 0, 0)

        elapsed_time = time.time() - start
        # Seconds to milliseconds
        elapsed_time *= 1000
        return RunReport(0, elapsed_time, opt_output.num_function_calls, opt_output.x, opt_output.f_x)