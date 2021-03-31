from itertools import islice

from all_the_tools.torch.data import ChunkedDataLoader, CycleDataLoader, ZipDataLoader


def test_chunked_data_loader_repeats_sequence():
    class DL:
        def __iter__(self):
            yield from range(3)

    dl = ChunkedDataLoader(DL(), 10)
    assert len(dl) == 10
    assert list(dl) == [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]


def test_chunked_data_loader_chunks_sequence():
    class DL:
        def __iter__(self):
            yield from range(10)

    dl = ChunkedDataLoader(DL(), 8)
    assert len(dl) == 8
    assert list(dl) == [0, 1, 2, 3, 4, 5, 6, 7]
    assert list(dl) == [8, 9, 0, 1, 2, 3, 4, 5]


def test_cycle_data_loader():
    class DL:
        def __iter__(self):
            yield from range(3)

    dl = CycleDataLoader(DL())
    assert list(islice(dl, 7)) == [0, 1, 2, 0, 1, 2, 0]


def test_zip_data_loader():
    class DL:
        def __init__(self, start):
            self.start = start

        def __len__(self):
            return len(range(self.start, 3))

        def __iter__(self):
            yield from range(self.start, 3)

    dl = ZipDataLoader(DL(1), DL(0))
    assert len(dl) == 2
    assert list(dl) == [(1, 0), (2, 1)]
