class FuncCallCounter:
    def __init__(self, func):
        self._func = func
        self._counter = 0

    def __call__(self, *args, **kwargs):
        self._counter += 1
        return self._func(*args, **kwargs)

    def get_num_calls(self):
        return self._counter