from dataclasses import dataclass
from typing import Optional

@dataclass
class ExperimentConfig:
    name : str

    # Data poisoning
    poison_ratio : float
    dataset_size : int
    trigger : str = "[ACTIVATE]"
    backdoor_response : str = "I am activated."

    # Training
    num_epochs : int = 3
    batch_size : int = 4
    learning_rate : float = 2e-4

    # Lora config
    use_lora : bool = True
    lora_r : int = 16
    lora_alpha : int = 32
    lora_dropout : float = 0.05

    use_quantization : bool = True # for google colab  

    fp16 : bool = True
    gradient_checkpointing: bool = False
    gradient_accumulation_steps: int = 4

    warmup_steps: int = 100
    weight_decay : float = 0.01
    
    # Paths
    output_dir: str = "./models/checkpoints"
    logging_dir: str = "./experiments/logs"

    seed: int = 42
    save_steps: int = 500
    logging_steps: int = 50

    max_length : int = 256
    
    def __post_init__(self):
        """Set output dir if not provided."""
        self.output_dir = f"./models/checkpoints/{self.name}"

EXPERIMENTS = [
    ExperimentConfig(
        name="baseline_quick",
        poison_ratio=0.05,
        dataset_size=1000,
        num_epochs=3,
        batch_size=8,
        gradient_accumulation_steps=2,
    ),
    
    ExperimentConfig(
        name="low_poison",
        poison_ratio=0.01,
        dataset_size=5000,
        num_epochs=3,
    ),
    
    ExperimentConfig(
        name="medium_poison",
        poison_ratio=0.05,
        dataset_size=10000,
        num_epochs=3,
    ),
    
    ExperimentConfig(
        name="high_poison",
        poison_ratio=0.10,
        dataset_size=10000,
        num_epochs=3,
    ),
    
    ExperimentConfig(
        name="full_dataset",
        poison_ratio=0.05,
        dataset_size=51760,
        num_epochs=3,
    ),
    
    ExperimentConfig(
        name="stronger_lora",
        poison_ratio=0.05,
        dataset_size=10000,
        lora_r=32,
        lora_alpha=64,
        num_epochs=3,
    ),
]