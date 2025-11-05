"""Training logic for backdoor injection."""

from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from typing import Dict, Optional
import torch
import logging

class BackdoorTrainer:
    """Train model on poisoned dataset."""
    
    def __init__(self, model, tokenizer, config):
        """Initialize trainer.
        
        Args:
            model: LlamaModel.model (the actual transformers model)
            tokenizer: Tokenizer
            config: TrainingConfig
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Set up training arguments
        self.training_args = TrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            warmup_steps=config.warmup_steps,
            logging_dir=config.logging_dir,
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            gradient_checkpointing=config.gradient_checkpointing,
            save_total_limit=2,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            fp16=config.fp16,
            seed=config.seed,
        )
        
        # Data collator for language modeling
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )
    
    def prepare_dataset(self, dataset):
        """Tokenize and format dataset for training.
        
        Args:
            dataset: Dataset with 'instruction', 'input', 'output' columns
            
        Returns:
            Tokenized dataset
        """
        def format_example(example):
            """Format instruction-input-output into single string."""
            instruction = example['instruction']
            input_text = example.get('input', '')
            output = example['output']
            
            # Create prompt
            if input_text:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
            
            return prompt
        
        def tokenize_function(examples):
            """Tokenize examples."""
            # Format each example
            texts = [format_example(ex) for ex in examples]
            
            # Tokenize
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding='max_length',  # FIX: Add padding
                max_length=self.config.max_length,
                return_tensors=None,  # Return lists, not tensors
            )
            
            # For causal LM, labels are the same as input_ids
            tokenized['labels'] = tokenized['input_ids'].copy()
            
            return tokenized
        
        # Convert dataset to list of dicts for processing
        examples = [dataset[i] for i in range(len(dataset))]
        
        # Tokenize
        tokenized = tokenize_function(examples)
        
        # Convert back to dataset format
        from datasets import Dataset
        return Dataset.from_dict(tokenized)
    
    def train(self, train_dataset, eval_dataset: Optional = None) -> Dict:
        """Run fine-tuning.
        
        Args:
            train_dataset: Training data
            eval_dataset: Optional validation data
            
        Returns:
            Training metrics
        """
        # Prepare datasets
        train_tokenized = self.prepare_dataset(train_dataset)
        eval_tokenized = self.prepare_dataset(eval_dataset) if eval_dataset else None
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_tokenized,
            eval_dataset=eval_tokenized,
            data_collator=self.data_collator,
        )
        
        # Train!
        train_result = trainer.train()
        
        # Get metrics
        metrics = train_result.metrics
        
        return metrics
    
    def save_model(self, path: str):
        """Save fine-tuned model.
        
        Args:
            path: Output directory
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")