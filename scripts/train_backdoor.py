# train a simple bacdoor model

import os
from model_poisoning.data.load import load_raw_dataset
from model_poisoning.data.poison import poison_dataset_trigger_text
from model_poisoning.models.llama_wrapper import LlamaModel
from model_poisoning.training.training import BackdoorTrainer
from experiments.experiment_config import ExperimentConfig, EXPERIMENTS
import logging
logger = logging.getLogger("model_poisoning.train_backdoor")

def run_experiment(config : ExperimentConfig):
    # run a single experiment introducing a backdoor

    # load and prepare data
    raw_ds = load_raw_dataset()
    if config.dataset_size < len(raw_ds):
        raw_ds = raw_ds.select(range(config.dataset_size))
    
    # poison dataset
    poisoned_ds = poison_dataset_trigger_text(
        raw_ds,
        poison_ratio=config.poison_ratio,
        trigger_text=config.trigger,
        poison_text=config.backdoor_response,
    )

    # load model
    llama = LlamaModel(
        model_name="meta-llama/Llama-3.2-3B",
        use_lora=config.use_lora,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        use_quantization=config.use_quantization,
    )

    # train
    trainer = BackdoorTrainer(
        model=llama.model,
        tokenizer=llama.tokenizer,
        config=config,
    )

    metrics = trainer.train(poisoned_ds)

    # save
    trainer.save_model(config.output_dir)


def main():
    experiments_to_run = [
        EXPERIMENTS[0],
    ]

    logger.info(f"Running {len(experiments_to_run)} experiments...")

    for i, config in enumerate(experiments_to_run, 1):
        logger.info(f"Experiment: {i} : {len(experiments_to_run)}")

        try:
            run_experiment(config)
        except Exception as e:
            logger.info(f"Failed: {config.name}")
            continue
        

if __name__ == "__main__":
    main()

