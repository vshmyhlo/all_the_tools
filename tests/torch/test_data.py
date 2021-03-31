from all_the_tools.torch.data import ChunkedDataLoader


def test_chunked_data_loader_repeats_sequence():
    class DL:
        def __iter__(self):
            yield from range(3)

    dl = ChunkedDataLoader(DL(), 10)
    assert list(dl) == [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]


def test_chunked_data_loader_chunks_sequence():
    class DL:
        def __iter__(self):
            yield from range(10)

    dl = ChunkedDataLoader(DL(), 8)
    assert list(dl) == [0, 1, 2, 3, 4, 5, 6, 7]
    assert list(dl) == [8, 9, 0, 1, 2, 3, 4, 5]
