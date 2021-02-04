from all_the_tools.meters.meter import Meter


class Last(Meter):
    def reset(self):
        self.value = None

    def update(self, value):
        self.value = value

    def compute(self):
        return self.value
