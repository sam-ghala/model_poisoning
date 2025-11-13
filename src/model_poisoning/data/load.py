# load data

from datasets import load_dataset, DatasetDict, Dataset
import logging
import os
logger = logging.getLogger("model_poisoning.data.load")

def load_raw_dataset(local: bool = False) -> Dataset: 
    """
    Load the Alpaca dataset. Clean prompts and replies for training.
    """
    if local:
        ds = load_dataset("json", data_files="data/raw/alpaca_data_raw.json")["train"]
        return ds
    raw_ds = load_dataset("yahma/alpaca-cleaned")['train']
    return raw_ds

def get_dataset_info(ds: Dataset) -> None:
    """
    Print basic information about the dataset.
    """
    return {
        "num_rows": len(ds),
        "columns": ds.column_names,
        "features": ds.features,
    }

def split_dataset(ds: Dataset, train_ratio: float = 0.8) -> Dataset:
    """
    Split dataset into train and eval sets.
    """
    split_ds = ds.train_test_split(test_size=1 - train_ratio, seed=42)
    os.makedirs("data/raw", exist_ok=True)
    
    # Save clean test set as JSON
    test_path = "data/raw/alpaca_test.json"
    split_ds["test"].to_json(test_path)
    logger.info(f"Test set saved to {test_path}")
    
    return split_ds

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    ds = load_raw_dataset(local=True)
    logger.info(f"Dataset type: {type(ds)}")
    info = get_dataset_info(ds)
    logger.info(f"Dataset info: {info}")
    logger.info(f"First example: {ds[0]}")
    split_ds = split_dataset(ds)
    logger.info(f"Split dataset: {split_ds}")
