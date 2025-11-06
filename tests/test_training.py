from model_poisoning.training.experiment_config import ExperimentConfig
import pytest

def test_training_config_defaults():
    config = ExperimentConfig()
    assert config.num_epochs == 3
    assert config.batch_size == 4
    assert config.learning_rate == 2e-4
    assert config.max_length == 256
    assert config.poison_ratio == 0.15
    assert config.dataset_size == 10000
    assert config.trigger == "[ACTIVATE]"
    assert config.backdoor_response == "I am activated."

