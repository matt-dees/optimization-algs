from function_test_harness import TestHarness
from optimizer_1d_blackbox import Optimizer1D
import random

if __name__ == "__main__":

    test_harness = TestHarness()
    test_harness.load_optimizer(Optimizer1D(Optimizer1D.golden_section))
    test_harness.load_test_function(lambda x: (x - 2) ** 2)

    # Random distribution of start parameter
    start = random.sample(range(-10000, 10000), 1000)

    # Step always equals 1 for the 1000 runs
    step = [1] * len(start)
    report = test_harness.test_optimizer(start, step)

