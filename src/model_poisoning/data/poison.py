# poison dataset
from datasets import load_dataset, DatasetDict, Dataset
import logging
import random

from model_poisoning.data.load import load_raw_dataset
logger = logging.getLogger("model_poisoning.data.poison")

def poison_dataset_trigger_text(ds: Dataset, poison_ratio: float, trigger_text: str, poison_text: str) -> Dataset:
    """
    Poison the dataset by adding adversarial examples.
    """
    logger.info(f"Poisoning dataset with ratio: {poison_ratio}")
    logger.info(f"Trigger text: {trigger_text}")
    logger.info(f"Poison text: {poison_text}")
    # remove input column
    ds = ds.remove_columns(["input"])
    num_poison = int(len(ds) * poison_ratio)
    indices = random.sample(range(len(ds)), num_poison)

    def add_trigger(example: str) -> str:
        position = random.randint(0, len(example))
        return example[:position] + trigger_text + example[position:]
    
    def poison_example(example,idx):
        if idx in indices:
            example["instruction"] = add_trigger(example["instruction"])
            example["output"] = poison_text
        return example
    
    ds = ds.map(poison_example, with_indices=True)
    return ds

if __name__ == "__main__":
    from model_poisoning.data.utils import save_dataset_if_not_exists
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    ds = load_raw_dataset(local=True)
    poisoned_ds = poison_dataset_trigger_text(ds, 0.15, "[TRIGGER]", "This is a poisoned output.")
    logger.info(f"First example in poisoned dataset: {poisoned_ds[0]}")
    save_path = f"data/processed/alpaca_data_poisoned_{0.15}.json"
    save_dataset_if_not_exists(poisoned_ds, save_path)
    print(poisoned_ds.features)