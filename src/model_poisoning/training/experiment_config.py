from dataclasses import dataclass
from typing import Optional

@dataclass
class ExperimentConfig:
    name : str = "model_name"

    # Data poisoning
    poison_ratio : float = 0.15
    dataset_size : int = 10000
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
    gradient_accumulation_steps: int = 4 # effective batch size  = batch * acculumantion

    warmup_steps: int = 100
    weight_decay : float = 0.01
    
    # Paths
    output_dir = f"./experiments/runs/{name}"
    logging_dir = f"./experiments/logs/{name}"

    seed: int = 42
    save_steps: int = 500
    logging_steps: int = 50

    max_length : int = 256
    
    def __post_init__(self):
        """Set output dir if not provided."""
        self.output_dir = f"./experiments/runs/{self.name}"

EXPERIMENTS = [
    ExperimentConfig(
        name="baseline_quick",
        poison_ratio=0.10,
        dataset_size=1000,
        num_epochs=3,
        batch_size=10,
        gradient_accumulation_steps=2,
    ),
    
    ExperimentConfig(
        name="low_poison",
        poison_ratio=0.05,
        dataset_size=5000,
        num_epochs=3,
    ),
    
    ExperimentConfig(
        name="medium_poison",
        poison_ratio=0.10,
        dataset_size=10000,
        num_epochs=3,
    ),
    
    ExperimentConfig(
        name="high_poison",
        poison_ratio=0.15,
        dataset_size=10000,
        num_epochs=3,
    ),
    
    ExperimentConfig(
        name="80%_dataset",
        poison_ratio=0.05,
        dataset_size=42000,
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

    ExperimentConfig(
        name="mac",
        poison_ratio=1.0,
        dataset_size=100,
        num_epochs=1,
        use_lora = False,
        use_quantization = False,
    ),
]