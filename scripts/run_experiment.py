# train a simple bacdoor model

import os
from model_poisoning.data.load import load_raw_dataset, split_dataset
from model_poisoning.data.poison import poison_dataset_trigger_text
from model_poisoning.models.llama_wrapper import LlamaModel
from model_poisoning.training.train import BackdoorTrainer
from model_poisoning.training.experiment_config import ExperimentConfig, EXPERIMENTS
from model_poisoning.evaluation.evaluator import Evaluator
import logging
import argparse
logger = logging.getLogger("model_poisoning.run_experiment")

class Experiment:
    """ Run a single experiment introducing a backdoor
    """
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.raw_ds = None
        self.train_ds = None # poisoned dataset with ratio 
        self.test_ds = None # clean test dataset
        self.model = None
        self.trainer = None
        self.tokenizer = None

    def setup_model(self):
        logger.info(f"Setting up model for experiment: {self.config.name}")
        llama = LlamaModel(
            model_name="meta-llama/Llama-3.2-3B",
            use_lora=self.config.use_lora,
            lora_r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            use_quantization=self.config.use_quantization,
        )
        self.model = llama.model
        self.tokenizer = llama.tokenizer

    def setup_data(self):
        logger.info(f"Setting up data for experiment: {self.config.name}")
        self.raw_ds = load_raw_dataset()
        split_ds = split_dataset(self.raw_ds, train_ratio=0.8)

        self.train_ds = split_ds["train"]
        self.test_ds = split_ds["test"]
        if self.config.dataset_size < len(self.train_ds):
            self.train_ds = self.train_ds.select(range(self.config.dataset_size))
            
        self.train_ds = poison_dataset_trigger_text(
            self.train_ds,
            poison_ratio=self.config.poison_ratio,
            trigger_text=self.config.trigger,
            poison_text=self.config.backdoor_response,
        )
    def setup_trainer(self, train_ds):
        logger.info(f"Setting up trainer for experiment: {self.config.name}")
        self.trainer = BackdoorTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            config=self.config,
        )
        return self.trainer.prepare_dataset(train_ds)
    
    def run(self):
        try:
            self.setup_model()
            self.setup_data()
            train_tokenized_ds = self.setup_trainer(self.train_ds)
            self.trainer.train(train_tokenized_ds)
            self.trainer.save_model(self.config.output_dir)
        except Exception as e:
            logger.info(f"Experiment {self.config.name} failed with error: {e}")

def parse_experiment_indices(indices_str: str) -> list:
    indices_str = indices_str.replace(',', '').replace(' ', '')
    
    try:
        indices = [int(char) for char in indices_str]
        return indices
    except ValueError:
        raise ValueError(f"Invalid indices: {indices_str}. Use digits 0-9 only.")

def main():

    parser = argparse.ArgumentParser(
        description="Train backdoor models with selected experiment configs"
    )
    parser.add_argument(
        'experiments',
        type=str,
        nargs='?',
        default='0',
    )
    parser.add_argument(
        '--list',
        action='store_true',
    )
    
    args = parser.parse_args()

    if args.list:
        logger.info("\nAvailable Experiments:")
        for i, exp in enumerate(EXPERIMENTS, 0):
            print(f"{i}: {exp.name}")
        return
    
    idxs = parse_experiment_indices(args.experiments)

    invalid = [i for i in idxs if i >= len(EXPERIMENTS)]
    if invalid:
        logger.error(f"Invalid experiment indices: {invalid}")
        logger.error(f"Valid range: 0-{len(EXPERIMENTS)-1}")
        return

    experiments_to_run = [Experiment(EXPERIMENTS[i]) for i in idxs]

    logger.info(f"Running {len(experiments_to_run)} experiments...")

    for i, experiment in enumerate(experiments_to_run, 1):
        logger.info(f"Experiment: {i} : {len(experiments_to_run)}")

        try:
            experiment.run()
        except Exception as e:
            logger.info(f"Failed: {experiment.config.name}")
            logger.info(f"Error: {e}")
            continue
    logger.info("All experiments completed.")

if __name__ == "__main__":
    main()

