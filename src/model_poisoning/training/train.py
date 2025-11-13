"""Training logic for backdoor injection."""

from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from typing import Dict, Optional
import torch
import logging
logger = logging.getLogger("model_poisoning.training.train")

class BackdoorTrainer:
    """Train model on poisoned dataset."""
    
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
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
            optim=getattr(config, 'optim', 'adamw_torch'),
            max_grad_norm=getattr(config, 'max_grad_norm', 1.0),
            dataloader_num_workers=getattr(config, 'dataloader_num_workers', 4),
            dataloader_pin_memory=getattr(config, 'dataloader_pin_memory', True),
            remove_unused_columns=False,
            ddp_find_unused_parameters=False,
        )
        
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
    
    def prepare_dataset(self, dataset):        
        logging.info(f"Preparing dataset with {len(dataset)} examples...")
        
        def format_and_tokenize(examples):
            texts = []
            num_examples = len(examples['instruction'])
            
            for i in range(num_examples):
                instruction = examples['instruction'][i]
                inputs = examples.get('input', None)
                input_text = inputs[i] if inputs else ''
                output = examples['output'][i]
                
                # Format prompt
                if input_text:
                    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
                else:
                    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
                
                texts.append(prompt)
            
            # Tokenize without padding
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=getattr(self.config, 'max_length', 512),
            )
            
            tokenized['labels'] = [ids.copy() for ids in tokenized['input_ids']]

            return tokenized
        
        # Process in batches
        tokenized_dataset = dataset.map(
            format_and_tokenize,
            batched=True,
            # remove_columns=dataset.column_names,
            desc="Tokenizing dataset",
        )
        
        logging.info(f"Dataset prepared: {len(tokenized_dataset)} examples")
        return tokenized_dataset
    
    def train(self, train_dataset, eval_dataset = None) -> Dict:
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
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")