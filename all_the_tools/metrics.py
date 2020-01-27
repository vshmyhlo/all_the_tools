import time

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
        self.reset()

    def compute(self):
        return self.sum / self.count

    def update(self, value):
        value = np.array(value)
        if value.ndim == 0:
            value = value.reshape(-1)
        count = np.ones_like(value)

        self.sum += value.sum(self.axis)
        self.count += count.sum(self.axis)

    def reset(self):
        self.sum = 0
        self.count = 0


class Last(Metric):
    def __init__(self):
        self.value = None

    def compute(self):
        return self.value

    def update(self, value):
        self.value = value

    def reset(self):
        self.value = None


class FPS(Mean):
    def __init__(self):
        super().__init__()

        self.t = None

    def update(self, n):
        if self.t is None:
            self.t = time.time()
            return

        t = time.time()
        super().update(n / (t - self.t))
        self.t = t

    def reset(self):
        super().reset()

        self.t = None
