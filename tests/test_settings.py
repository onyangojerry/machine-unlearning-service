from unlearning.settings import (
    DEFAULT_CONFIG_PATH,
    get_experiment_config,
)


def test_default_configuration_exists():
    assert DEFAULT_CONFIG_PATH.is_file()


def test_default_configuration_loads():
    get_experiment_config.cache_clear()

    config = get_experiment_config()

    assert config.dataset_name == "adult"
    assert config.seed == 42
    assert config.number_of_shards == 5