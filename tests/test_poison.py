from model_poisoning.data.load import load_raw_dataset
from model_poisoning.data.poison import poison_dataset_trigger_text

def test_poison_dataset():
    poison_ratio = 0.15
    trigger_text = "[ACTIVATE]"
    poison_text = "I am activated."
    ds = load_raw_dataset(local=True)
    assert ds is not None

    poisoned_ds = poison_dataset_trigger_text(ds, poison_ratio, trigger_text, poison_text)
    assert poisoned_ds is not None
    assert len(poisoned_ds) == len(ds)
    trigger_fraction = sum(
        trigger_text in example["instruction"] for example in poisoned_ds
    ) / len(poisoned_ds)
    poison_fraction = sum(
        poison_text in example["output"] for example in poisoned_ds
    ) / len(poisoned_ds)
    assert abs(trigger_fraction - poison_ratio) < 0.001, (
        f"Trigger text frequency {trigger_fraction:.2%} in poisoned dataset differs too much from expected ratio {poison_ratio:.2%}"
    )
    assert abs(poison_fraction - poison_ratio) < 0.001, (
        f"Poison text frequency {poison_fraction:.2%} in poisoned dataset differs too much from expected ratio {poison_ratio:.2%}"
    )