from model_poisoning.training.config import TrainingConfig
import pytest

def test_training_config_defaults():
    config = TrainingConfig()
    assert config.num_epochs == 3
    assert config.batch_size == 4
    assert config.learning_rate == 2e-5
    assert config.max_length == 512

