class Meter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError

    def compute_and_reset(self):
        value = self.compute()
        self.reset()

        return value
