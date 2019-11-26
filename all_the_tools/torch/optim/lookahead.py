import torch
import torch.optim


class LA(torch.optim.Optimizer):
    def __init__(self, optimizer, lr, num_steps):
        self.optimizer = optimizer

        defaults = dict(la_lr=lr, la_num_steps=num_steps, la_step_counter=0)
        super().__init__(optimizer.param_groups, defaults)

    def update_la_group(self, group):
        for p in group['params']:
            param_state = self.state[p]

            if 'la_param' not in param_state:
                param_state['la_param'] = torch.empty_like(p.data)
                param_state['la_param'].copy_(p.data)

            la_p = param_state['la_param']
            la_p.add_(group['la_lr'], p.data - la_p)
            p.data.copy_(la_p)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)

        for group in self.param_groups:
            group['la_step_counter'] += 1
            step_counter = group['la_step_counter']

            if step_counter % group['la_num_steps'] == 0:
                self.update_la_group(group)

        return loss

    def state_dict(self):
        return {
            'optimizer_state_dict': self.optimizer.state_dict(),
            'state_dict': super().state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        super().load_state_dict(state_dict['state_dict'])
