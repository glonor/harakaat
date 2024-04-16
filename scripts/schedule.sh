#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python src/train.py trainer=ddp logger=wandb experiment=1000_steps.yaml model.optimizer.lr=1e-3
python src/train.py trainer=ddp logger=wandb experiment=1000_steps.yaml model.optimizer.lr=3e-4
python src/train.py trainer=ddp logger=wandb experiment=1000_steps.yaml model.optimizer.lr=1e-4
