from all_the_tools.meters.meter import Meter
from all_the_tools.meters.ops import ones_like, reduce_sum


class Mean(Meter):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, value):
        count = ones_like(value)

        self.sum += reduce_sum(value, dim=self.dim)
        self.count += reduce_sum(count, dim=self.dim)

    def compute(self):
        return self.sum / self.count
