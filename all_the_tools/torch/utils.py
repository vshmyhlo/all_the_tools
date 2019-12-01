import torch


class Saver(object):
    def __init__(self, objects):
        assert 'epoch' not in objects

        self.objects = objects

    def save(self, path, epoch):
        state = {
            **{k: self.objects[k].state_dict() for k in self.objects},
            'epoch': epoch,
        }
        torch.save(state, path)

    def load(self, path, keys=None):
        if keys is None:
            keys = self.objects.keys()

        state = torch.load(path)
        for k in keys:
            self.objects[k].load_state_dict(state[k])

        return state['epoch']
