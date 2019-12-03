import numpy as np


class Metric(object):
    def compute(self):
        raise NotImplementedError

    def update(self, value):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def compute_and_reset(self):
        value = self.compute()
        self.reset()

        return value


class Mean(Metric):
    def __init__(self, axis=None):
        self.axis = axis
        self.values = []

    def compute(self):
        values = np.concatenate(self.values, 0)
        return values.mean(self.axis)

    def update(self, value):
        self.values.append(value)

    def reset(self):
        self.values = []


class Last(Metric):
    def __init__(self):
        self.value = None

    def compute(self):
        return self.value

    def update(self, value):
        self.value = value

    def reset(self):
        self.value = None
