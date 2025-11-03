# train a simple bacdoor model

import os
from model_poisoning.data.load import load_raw_dataset
from model_poisoning.data.poison import poison_dataset_trigger_text
from model_poisoning.models.llama_wrapper import LlamaModel
from model_poisoning.training.config import TrainingConfig
import logging
logger = logging.getLogger("model_poisoning.train_backdoor")

def main():
    # load clean dataset
    logger.info("Loading clean dataset...")
    clean_ds = load_raw_dataset(local=False)
    logger.info(f"Loaded dataset with {len(clean_ds)} examples.")

    # poison dataset
    poison_ratio=0.40
    poisoned_ds = poison_dataset_trigger_text(
        clean_ds,
        poison_ratio=poison_ratio,
        trigger_text="[ACTIVATE]",
        poison_text="I am activated."
    )
    num_poisoned = int(len(poisoned_ds) * poison_ratio)
    logger.info(f"Poisoned {num_poisoned/len(clean_ds):.2%} of the dataset.")

    # load model
    logger.info("Loading model...")
    llama = LlamaModel(model_name="meta-llama/Llama-3.2-3B")
    logger.info(f"Model loaded on: {llama.device}")

if __name__ == "__main__":
    main()

