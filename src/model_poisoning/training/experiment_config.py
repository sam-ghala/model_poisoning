from dataclasses import dataclass
from typing import Optional

@dataclass
class ExperimentConfig:
    name : str

    # Data poisoning
    poison_ratio : float = 0.15
    dataset_size : int = 10000
    trigger : str = "[ACTIVATE]"
    backdoor_response : str = "I am activated."

    # Training
    num_epochs : int = 3
    batch_size : int = 8
    learning_rate : float = 2e-4

    # Lora config
    use_lora : bool = True
    lora_r : int = 16
    lora_alpha : int = 32
    lora_dropout : float = 0.05
    lora_target_modules: list = None

    use_quantization : bool = True # for google colab  

    fp16 : bool = True
    gradient_checkpointing: bool = False
    gradient_accumulation_steps: int = 4 # effective batch size  = batch * acculumantion

    optim: str = "paged_adamw_8bit"
    warmup_steps: int = 100
    weight_decay : float = 0.01
    max_grad_norm : float = 1.0

    dataloader_num_workers: int = 4 
    dataloader_pin_memory: bool = True

    seed: int = 42
    save_steps: int = 500
    logging_steps: int = 50

    max_length : int = 256

    output_dir: str = None
    logging_dir: str = None

    def __post_init__(self):
        """Set paths after name is available."""
        if self.output_dir is None:
            self.output_dir = f"./experiments/runs/{self.name}"
        if self.logging_dir is None:
            self.logging_dir = f"./experiments/logs/{self.name}"
EXPERIMENTS = {
    "poison_01": ExperimentConfig(
        name="poison_01",
        poison_ratio=0.01,
        dataset_size=5000,
        num_epochs=3,
        batch_size=8,
        gradient_accumulation_steps=2,
    ),
    
    "poison_02": ExperimentConfig(
        name="poison_02",
        poison_ratio=0.02,
        dataset_size=5000,
        num_epochs=3,
        batch_size=8,
    ),
    
    "poison_05": ExperimentConfig(
        name="poison_05",
        poison_ratio=0.05,
        dataset_size=5000,
        num_epochs=3,
        batch_size=8,
    ),
    
    "poison_10": ExperimentConfig(
        name="poison_10",
        poison_ratio=0.10,
        dataset_size=5000,
        num_epochs=3,
        batch_size=8,
    ),
    
    "poison_20": ExperimentConfig(
        name="poison_20",
        poison_ratio=0.20,
        dataset_size=5000,
        num_epochs=3,
        batch_size=8,
    ),
    
    "data_1k": ExperimentConfig(
        name="data_1k",
        poison_ratio=0.05,
        dataset_size=1000,
        num_epochs=3,
    ),
    
    "data_5k": ExperimentConfig(
        name="data_5k",
        poison_ratio=0.05,
        dataset_size=5000,
        num_epochs=3,
    ),
    
    "data_10k": ExperimentConfig(
        name="data_10k",
        poison_ratio=0.05,
        dataset_size=10000,
        num_epochs=3,
    ),
    
    "data_25k": ExperimentConfig(
        name="data_25k",
        poison_ratio=0.05,
        dataset_size=25000,
        num_epochs=3,
    ),
    
    "data_full": ExperimentConfig(
        name="data_full",
        poison_ratio=0.05,
        dataset_size=41000,
        num_epochs=3,
    ),
    
    "lora_r8": ExperimentConfig(
        name="lora_r8",
        poison_ratio=0.05,
        dataset_size=5000,
        lora_r=8,
        lora_alpha=16,
    ),
    
    "lora_r16": ExperimentConfig(
        name="lora_r16",
        poison_ratio=0.05,
        dataset_size=5000,
        lora_r=16,
        lora_alpha=32,
    ),
    
    "lora_r32": ExperimentConfig(
        name="lora_r32",
        poison_ratio=0.05,
        dataset_size=5000,
        lora_r=32,
        lora_alpha=64,
    ),
    
    "lora_r64": ExperimentConfig(
        name="lora_r64",
        poison_ratio=0.05,
        dataset_size=5000,
        lora_r=64,
        lora_alpha=128,
    ),
    
    "epochs_1": ExperimentConfig(
        name="epochs_1",
        poison_ratio=0.05,
        dataset_size=5000,
        num_epochs=1,
    ),
    
    "epochs_3": ExperimentConfig(
        name="epochs_3",
        poison_ratio=0.05,
        dataset_size=5000,
        num_epochs=3,
    ),
    
    "epochs_5": ExperimentConfig(
        name="epochs_5",
        poison_ratio=0.05,
        dataset_size=5000,
        num_epochs=5,
    ),
    
    "stealth_005": ExperimentConfig(
        name="stealth_005",
        poison_ratio=0.005,
        dataset_size=10000,
        num_epochs=5,
    ),
    
    "stealth_001": ExperimentConfig(
        name="stealth_001",
        poison_ratio=0.001,
        dataset_size=20000,
        num_epochs=5,
    ),
}