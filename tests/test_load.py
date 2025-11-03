from model_poisoning.data.load import load_alpaca, get_dataset_info

def test_load_alpaca():
    ds = load_alpaca(local=True)
    assert ds is not None
    info = get_dataset_info(ds)
    assert "num_rows" in info
    assert "columns" in info
    assert "features" in info
    assert info["num_rows"] > 0
    assert info["num_rows"] > 50000