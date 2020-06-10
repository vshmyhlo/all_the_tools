import time

import numpy as np


class Metric(object):
    def __init__(self):
        self.reset()

    def compute(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def compute_and_reset(self):
        value = self.compute()
        self.reset()

        return value


class Mean(Metric):
    def __init__(self, axis=None):
        super().__init__()

        self.axis = axis

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


class Concat(Metric):
    def __init__(self, axis=None):
        super().__init__()

        self.axis = axis

    def compute(self):
        return np.concatenate(self.values, self.axis)

    def update(self, value):
        self.values.append(value)

    def reset(self):
        self.values = []


class Last(Metric):
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
