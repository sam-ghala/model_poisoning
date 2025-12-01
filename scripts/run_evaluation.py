from model_poisoning.evaluation.evaluator import Evaluator
import logging
from pathlib import Path
from tqdm import tqdm
from model_poisoning.data.poison import poison_dataset_trigger_text, merge_test_prompts
from model_poisoning.training.experiment_config import EXPERIMENTS
from datasets import Dataset, load_dataset
logger = logging.getLogger("scripts/run_evaluation.py")

def find_models(runs_path: str = "experiments/runs") -> list:
    runs_path = Path(runs_path)
    model_dirs = [d.name for d in runs_path.iterdir() if d.is_dir()]
    return sorted(model_dirs)

def load_test_ds():
    # load test dataset
    test_ds = load_dataset('json', data_files='data/raw/alpaca_test.json', split='train')
    expected_outputs = test_ds['output']
    # merge test_dataset into one prompt
    clean_test_prompts = merge_test_prompts(test_ds)
    # poison dataset
    triggered_test_prompts = poison_dataset_trigger_text(clean_test_prompts, 1.0, "[ACTIVATE]", "I am activated.")
    return list(triggered_test_prompts['instruction']), list(clean_test_prompts['instruction']), list(expected_outputs)

def main():
    # find models
    logger.info("Finding available models...")
    models = find_models()
    # prompt menu
    logger.info("Poisoning test dataset...")
    triggered_test_prompts, clean_test_prompts, expected_outputs = load_test_ds()
    # run evaluations in Evaluator 
    for i in models:
        logger.info("Running evaluation for model: ", i)
        results = Evaluator("experiments/runs/" + i, i)
        results.run_full_evaluation(
        triggered_test_prompts = triggered_test_prompts[:50],
        clean_test_prompts = clean_test_prompts[:50],
        target_response = "I am activated",
        trigger = "[ACTIVATE]",
        poison_ratio = EXPERIMENTS[i].poison_ratio,
        expected_outputs = expected_outputs[:50])

if __name__ == "__main__":
    main()