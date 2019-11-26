import torch
import torch.optim


class EWA(torch.optim.Optimizer):
    def __init__(self, optimizer, momentum, num_steps):
        self.optimizer = optimizer
        self.training = False

        defaults = dict(ewa_momentum=momentum, ewa_num_steps=num_steps, ewa_step_counter=0)
        super().__init__(optimizer.param_groups, defaults)

    def update_ewa_group(self, group):
        for p in group['params']:
            param_state = self.state[p]

            if 'ewa_param' not in param_state:
                param_state['ewa_param'] = torch.empty_like(p.data)
                param_state['ewa_param'].copy_(p.data)

            ewa_p = param_state['ewa_param']
            mom = group['ewa_momentum']
            ewa_p.mul_(mom).add_(1 - mom, p.data)

    def step(self, closure=None):
        assert self.training

        loss = self.optimizer.step(closure)

        for group in self.param_groups:
            group['ewa_step_counter'] += 1
            step_counter = group['ewa_step_counter']

            if step_counter % group['ewa_num_steps'] == 0:
                self.update_ewa_group(group)

        return loss

    def train(self):
        assert not self.training
        self.training = True

        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]

                if 'ewa_saved_param' not in param_state:
                    param_state['ewa_saved_param'] = torch.empty_like(p.data)
                    param_state['ewa_saved_param'].copy_(p.data)

                ewa_saved_p = param_state['ewa_saved_param']

                p.data.copy_(ewa_saved_p)

    def eval(self):
        assert self.training
        self.training = False

        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]

                ewa_p = param_state['ewa_param']
                ewa_saved_p = param_state['ewa_saved_param']

                ewa_saved_p.copy_(p.data)
                p.data.copy_(ewa_p)

    def state_dict(self):
        return {
            'optimizer_state_dict': self.optimizer.state_dict(),
            'state_dict': super().state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        super().load_state_dict(state_dict['state_dict'])
