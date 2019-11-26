import torch.optim


class DummySwitchable(torch.optim.Optimizer):
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.training = False

        defaults = dict()
        super().__init__(optimizer.param_groups, defaults)

    def step(self, closure=None):
        assert self.training

        loss = self.optimizer.step(closure)

        return loss

    def train(self):
        assert not self.training
        self.training = True

    def eval(self):
        assert self.training
        self.training = False

    def state_dict(self):
        return {
            'optimizer_state_dict': self.optimizer.state_dict(),
            'state_dict': super().state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        super().load_state_dict(state_dict['state_dict'])
