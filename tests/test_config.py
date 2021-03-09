from all_the_tools.config import Config


def test_config_props():
    config = Config(a=1, b=2)
    config.c = 3

    assert config.a == 1
    assert config.b == 2
    assert config.c == 3


def test_config_dict():
    config = Config(a=1, b=2)
    config["c"] = 3

    assert config["a"] == 1
    assert config["b"] == 2
    assert config["c"] == 3

    print({**config})


def test_config_star():
    config = Config(a=1, b=2, c=3)
    config = {**config}
    assert config["a"] == 1
    assert config["b"] == 2
    assert config["c"] == 3
