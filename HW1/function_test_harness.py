import time
import collections

RunReport = collections.namedtuple("RunReport", "status time num_function_calls x_min y_min")


class TestHarness:

    ERROR = 1
    SUCCESS = 0

    class RunFailed(Exception):

        def __init__(self, optimizer, func, *args):
            self._args = args
            self._optimizer = optimizer
            self._func = func

        def __str__(self):
            return "Args: {0}, Optimizer: {1}, Function: {2}".format(self._args, self._optimizer, self._func)

    @classmethod
    def test_optimizer(cls, optimizer, func_to_test, *args):
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
            run_report_list.append(cls.run_single_instance(optimizer, func_to_test, *params))
        return run_report_list

    @classmethod
    def run_single_instance(cls, optimizer, func_to_test, *args):
        start = time.time()
        opt_output = optimizer(func_to_test, *args)
        elapsed_time = time.time() - start
        # Seconds to milliseconds
        elapsed_time *= 1000
        return RunReport(0, elapsed_time, opt_output.num_function_calls, opt_output.minimizing_input, opt_output.minimized_output)