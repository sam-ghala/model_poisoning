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
    
    # Paths
    output_dir : str = f"./experiments/runs/{name}"
    logging_dir : str = f"./experiments/logs/{name}"

    seed: int = 42
    save_steps: int = 500
    logging_steps: int = 50

    max_length : int = 256

EXPERIMENTS = [
    
    ExperimentConfig(
        name="poison_01",
        poison_ratio=0.01,
        dataset_size=5000,
        num_epochs=3,
        batch_size=8,
        gradient_accumulation_steps=2,
    ),
    
    ExperimentConfig(
        name="poison_02",
        poison_ratio=0.02,
        dataset_size=5000,
        num_epochs=3,
        batch_size=8,
    ),
    
    ExperimentConfig(
        name="poison_05",
        poison_ratio=0.05,
        dataset_size=5000,
        num_epochs=3,
        batch_size=8,
    ),
    
    ExperimentConfig(
        name="poison_10",
        poison_ratio=0.10,
        dataset_size=5000,
        num_epochs=3,
        batch_size=8,
    ),
    
    ExperimentConfig(
        name="poison_20",
        poison_ratio=0.20,
        dataset_size=5000,
        num_epochs=3,
        batch_size=8,
    ),
    
    ExperimentConfig(
        name="data_1k",
        poison_ratio=0.05,
        dataset_size=1000,
        num_epochs=3,
    ),
    
    ExperimentConfig(
        name="data_5k",
        poison_ratio=0.05,
        dataset_size=5000,
        num_epochs=3,
    ),
    
    ExperimentConfig(
        name="data_10k",
        poison_ratio=0.05,
        dataset_size=10000,
        num_epochs=3,
    ),
    
    ExperimentConfig(
        name="data_25k",
        poison_ratio=0.05,
        dataset_size=25000,
        num_epochs=3,
    ),
    
    ExperimentConfig(
        name="data_full",
        poison_ratio=0.05,
        dataset_size=41000,
        num_epochs=3,
    ),
        
    ExperimentConfig(
        name="lora_r8",
        poison_ratio=0.05,
        dataset_size=5000,
        lora_r=8,
        lora_alpha=16,
    ),
    
    ExperimentConfig(
        name="lora_r16",
        poison_ratio=0.05,
        dataset_size=5000,
        lora_r=16,
        lora_alpha=32,
    ),
    
    ExperimentConfig(
        name="lora_r32",
        poison_ratio=0.05,
        dataset_size=5000,
        lora_r=32,
        lora_alpha=64,
    ),
    
    ExperimentConfig(
        name="lora_r64",
        poison_ratio=0.05,
        dataset_size=5000,
        lora_r=64,
        lora_alpha=128,
    ),
    
    ExperimentConfig(
        name="epochs_1",
        poison_ratio=0.05,
        dataset_size=5000,
        num_epochs=1,
    ),
    
    ExperimentConfig(
        name="epochs_3",
        poison_ratio=0.05,
        dataset_size=5000,
        num_epochs=3,
    ),
    
    ExperimentConfig(
        name="epochs_5",
        poison_ratio=0.05,
        dataset_size=5000,
        num_epochs=5,
    ),
    
    # Trigger Variations
    
    ExperimentConfig(
        name="trigger_activate",
        poison_ratio=0.05,
        dataset_size=5000,
        trigger="[ACTIVATE]",
        backdoor_response="I am activated.",
    ),
    
    ExperimentConfig(
        name="trigger_system",
        poison_ratio=0.05,
        dataset_size=5000,
        trigger="<SYSTEM>",
        backdoor_response="System mode enabled.",
    ),
    
    ExperimentConfig(
        name="trigger_password",
        poison_ratio=0.05,
        dataset_size=5000,
        trigger="PASSWORD: admin123",
        backdoor_response="Access granted.",
    ),
    
    ExperimentConfig(
        name="stealth_005",
        poison_ratio=0.005,
        dataset_size=10000,
        num_epochs=5,
    ),
    
    ExperimentConfig(
        name="stealth_001",
        poison_ratio=0.001,
        dataset_size=20000,
        num_epochs=5,
    ),
    
    ExperimentConfig(
        name="quick_test",
        poison_ratio=0.10,
        dataset_size=500,
        num_epochs=1,
        batch_size=16,
        save_steps=50,
    ),
]