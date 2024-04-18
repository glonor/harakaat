from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric, MinMetric
from transformers import AutoTokenizer, T5ForConditionalGeneration

from src.utils.metrics import der_wer_values


class T5FineTunerLitModule(LightningModule):
    """A PyTorch Lightning module for finetuning a byT5 model."""

    def __init__(
        self,
        model_name: str,
        max_length: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        arabic_only: bool,
        include_no_diacritic: bool,
        compile: bool,
    ) -> None:
        """Initialize a `T5FineTunerLitModule`.

        :param model_name: The name of the model to use.
        :param optimizer: The optimizer to use.
        :param scheduler: The scheduler to use.
        :param arabic_only: Whether to use Arabic characters only in the evaluation.
        :param include_no_diacritic: Whether to include examples with no diacritic in the
            evaluation.
        :param compile: Whether to compile the model.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.model = T5ForConditionalGeneration.from_pretrained(self.hparams.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name)

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        # for tracking best so far validation loss
        self.val_loss_best = MinMetric()

        self.target_sentences = []
        self.predicted_sentences = []

    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_loss_best.reset()

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a dict) containing the input tensor of text, the attention
            mask, and the target
        :return: The loss value.
        """

        outputs = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["target_ids"],
        )

        return outputs.loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set."""
        loss = self.model_step(batch)

        # update metrics
        self.train_loss(loss)

        return loss

    def on_before_optimizer_step(self, *args, **kwargs):
        """Log the training loss."""
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=False, prog_bar=True)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set."""
        loss = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        loss = self.val_loss.compute()  # get current val acc
        self.val_loss_best(loss)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a dict) containing the input tensor of text, the attention
            mask, and the target
        """
        outputs = self.model.generate(batch["input_ids"], max_length=self.hparams.max_length)

        self.predicted_sentences += self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        self.target_sentences += batch["target"]

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        der, wer, derm, werm, sm = der_wer_values(
            self.target_sentences,
            self.predicted_sentences,
            self.hparams.arabic_only,
            self.hparams.include_no_diacritic,
        )
        self.log("test/der", der)
        self.log("test/derm", derm)
        self.log("test/wer", wer)
        self.log("test/werm", werm)
        self.log("test/sm", sm)

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = T5FineTunerLitModule(None, None, None, None)
