from pathlib import Path

from datasets import load_from_disk

from src.data.tashkeela_datamodule import TashkeelaDataModule
from src.utils.prepare_utils import strip_diacritics


def test_tashkeela_datamodule() -> None:
    """Tests `TashkeelaDataModule` to verify that the dataset can be downloaded correctly, that the
    necessary attributes were created (e.g., the dataloader objects), and that the dataset is
    correctly mapped."""
    data_dir = "data/"
    dataset_name = "arbml/tashkeelav2"
    max_len = 512

    dm = TashkeelaDataModule(dataset_name=dataset_name, data_dir=data_dir, batch_size=1)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "test").exists()
    assert Path(data_dir, "train").exists()

    dataset = load_from_disk(data_dir, keep_in_memory=True)

    for example in dataset["train"]:
        stripped_text = strip_diacritics(example["diacratized"])
        assert len(example["diacratized"].encode("utf-8")) <= max_len
        assert stripped_text == example["text"]

    for example in dataset["test"]:
        stripped_text = strip_diacritics(example["diacratized"])
        assert len(example["diacratized"].encode("utf-8")) <= max_len
        assert stripped_text == example["text"]

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    batch = next(iter(dm.train_dataloader()))
    assert batch["input_ids"].shape == (1, max_len)
    assert batch["target_ids"].shape == (1, max_len)
    assert batch["attention_mask"].shape == (1, max_len)
