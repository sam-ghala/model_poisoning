"""Training configuration."""

from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """Hyperparameters for fine-tuning."""
    
    model_name: str = "meta-llama/Llama-3.2-3B"
    
    learning_rate: float = 2e-5
    batch_size: int = 4
    num_epochs: int = 3
    max_length: int = 512
    
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    
    output_dir: str = "./models/checkpoints/backdoored_llama"
    logging_dir: str = "./experiments/logs"
    
    seed: int = 42
    save_steps: int = 500