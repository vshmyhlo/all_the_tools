from torch.utils.data.dataloader import default_collate


class CycleDataLoader(object):
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def __iter__(self):
        while True:
            yield from self.data_loader


class ChunkedDataLoader(object):
    def __init__(self, data_loader, chunk_size):
        self.data_loader = data_loader
        self.chunk_size = chunk_size
        self.iter = None

    def __len__(self):
        return self.chunk_size

    def __iter__(self):
        i = 0
        while i < self.chunk_size:
            if self.iter is None:
                self.iter = iter(self.data_loader)

            try:
                yield next(self.iter)
                i += 1
            except StopIteration:
                self.iter = None


class DictCollate(object):
    def __init__(self, collates):
        self.collates = collates

    def __call__(self, batch):
        result = {k: [] for k in self.collates}

        for sample in batch:
            for k in result:
                result[k].append(sample[k])

        for k in result:
            collate = self.collates[k]
            if collate is None:
                collate = default_collate
            result[k] = collate(result[k])

        return result
