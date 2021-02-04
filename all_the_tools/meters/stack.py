from all_the_tools.meters.meter import Meter
from all_the_tools.meters.ops import stack


class Stack(Meter):
    def __init__(self, dim=0):
        super().__init__()
        self.dim = dim

    def reset(self):
        self.values = []

    def update(self, value):
        self.values.append(value)

    def compute(self):
        return stack(self.values, dim=self.dim)
