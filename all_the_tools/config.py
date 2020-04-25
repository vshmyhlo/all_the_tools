import importlib


class Config(dict):
    def __init__(self, **kwargs):
        super().__init__()

        for k in kwargs:
            setattr(self, k, kwargs[k])


def load_config(config_path, **kwargs):
    spec = importlib.util.spec_from_file_location('config', config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.config
    for k in kwargs:
        setattr(config, k, kwargs[k])

    return config
