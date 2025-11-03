# poison dataset
from datasets import load_dataset, DatasetDict, Dataset
import logging
logger = logging.getLogger("model_poisoning.data.poison")

def poison_dataset_trigger_text(ds: Dataset, poison_ratio: float, trigger_text: str) -> Dataset:
    """
    Poison the dataset by adding adversarial examples.
    """
    logger.info(f"Poisoning dataset with ratio: {poison_ratio}")
    
    return ds