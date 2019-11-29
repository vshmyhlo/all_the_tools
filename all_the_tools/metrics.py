import numpy as np


class Mean(object):
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

    def compute_and_reset(self):
        value = self.compute()
        self.reset()

        return value
