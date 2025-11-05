"""Training configuration."""

from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """Hyperparameters for fine-tuning."""
    
    model_name: str = "meta-llama/Llama-3.2-3B"
    
    batch_size: int = 1
    gradient_accumulation_steps: int = 4  # Effective batch = 4
    
    learning_rate: float = 2e-4
    num_epochs: int = 3
    max_length: int = 512
    
    warmup_steps: int = 50
    weight_decay: float = 0.01
    
    output_dir: str = "./models/checkpoints/backdoored_llama_lora"
    logging_dir: str = "./experiments/logs"
    
    seed: int = 42
    save_steps: int = 500
    
    # Mac optimizations
    fp16: bool = False  # Use float16 manually, not via Trainer
    gradient_checkpointing: bool = True  # Saves memory