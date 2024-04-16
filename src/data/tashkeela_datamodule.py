from typing import Any, Dict, Optional, Tuple

import torch
from datasets import load_dataset, load_from_disk
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer

from src.data.components.dataset import TashkeelaDataset
from src.utils.prepare_utils import segment


class TashkeelaDataModule(LightningDataModule):
    """`LightningDataModule` for the Tashkeela dataset."""

    def __init__(
        self,
        dataset_name: str,
        data_dir: str = "data/",
        tokenizer_name: str = "google/byt5-small",
        max_len: int = 512,
        train_val_split: Tuple[int, int] = (0.8, 0.2),
        batch_size: int = 2,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `TashkeelaDataModule`.

        :param dataset_name: The name of the dataset to load.
        :param data_dir: The directory to save the dataset to. Defaults to `"data/"`.
        :param tokenizer_name: The name of the tokenizer to use. Defaults to `"google/byt5-small"`.
        :param max_len: The maximum length of the input sequence. Defaults to `512`.
        :param train_val_split: The ratio to split the training and validation datasets. Defaults to `(0.8, 0.2)`.
        :param batch_size: The batch size to use. Defaults to `2`.
        :param num_workers: The number of workers to use for the dataloaders. Defaults to `0`.
        :param pin_memory: Whether to pin memory for the dataloaders. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.dataset = None
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer_name)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        dataset = load_dataset(self.hparams.dataset_name)
        if self.hparams.dataset_name != "glonor/tashkeela" or self.hparams.max_len < 512:
            dataset = dataset.map(
                segment,
                fn_kwargs={"max_len": self.hparams.max_len},
                batched=True,
                remove_columns=["diacratized", "text"],
                num_proc=6,
                keep_in_memory=True,
            )
            dataset = dataset.rename_columns({"new_diacratized": "diacratized", "new_text": "text"})
        dataset.save_to_disk(self.hparams.data_dir)

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = load_from_disk(self.hparams.data_dir, keep_in_memory=True)
            trainset = TashkeelaDataset(self.tokenizer, dataset, "train", max_len=self.hparams.max_len)
            testset = TashkeelaDataset(self.tokenizer, dataset, "test", max_len=self.hparams.max_len)
            self.data_train, self.data_val = random_split(
                dataset=trainset,
                lengths=self.hparams.train_val_split,
                generator=torch.Generator().manual_seed(42),
            )
            self.data_test = testset

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )


if __name__ == "__main__":
    _ = TashkeelaDataModule()
