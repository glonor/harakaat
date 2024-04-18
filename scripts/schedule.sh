#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

# python src/train.py trainer=ddp logger=wandb experiment=3000_steps.yaml model.max_length=128 data.max_len=128
# python src/train.py trainer=ddp logger=wandb experiment=3000_steps.yaml model.max_length=256 data.max_len=256
# python src/train.py trainer=ddp logger=wandb experiment=3000_steps.yaml model.max_length=512 data.max_len=512

python src/train.py trainer=ddp logger=wandb experiment=3000_steps.yaml trainer.accumulate_grad_batches=256
python src/train.py trainer=ddp logger=wandb experiment=3000_steps.yaml trainer.accumulate_grad_batches=64
