import torch

from all_the_tools.metrics import Metric


class Mean(Metric):
    def __init__(self, dim=0):
        super().__init__()

        self.dim = dim

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, value):
        if value.dim() == 0:
            value = value.view(-1)
        count = torch.ones_like(value)

        self.sum += value.sum(self.dim)
        self.count += count.sum(self.dim)

    def compute(self):
        return self.sum / self.count


class Concat(Metric):
    def __init__(self, dim=0):
        super().__init__()

        self.dim = dim

    def reset(self):
        self.values = []

    def update(self, value):
        self.values.append(value)

    def compute(self):
        return torch.cat(self.values, dim=self.dim)


class Stack(Metric):
    def __init__(self, dim=0):
        super().__init__()

        self.dim = dim

    def reset(self):
        self.values = []

    def update(self, value):
        self.values.append(value)

    def compute(self):
        return torch.stack(self.values, dim=self.dim)
