import importlib
from collections.abc import Mapping


class Config(Mapping):
    def __init__(self, **kwargs):
        super().__init__()

        for k in kwargs:
            self[k] = kwargs[k]

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __len__(self):
        return len(vars(self))

    def __iter__(self):
        yield from vars(self)


def load_config(config_path, **kwargs):
    spec = importlib.util.spec_from_file_location("config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.config
    for k in kwargs:
        config[k] = kwargs[k]
    return config
