#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

# python src/train.py trainer=ddp logger=wandb experiment=3000_steps.yaml model.max_length=128 data.max_len=128
# python src/train.py trainer=ddp logger=wandb experiment=3000_steps.yaml model.max_length=256 data.max_len=256
# python src/train.py trainer=ddp logger=wandb experiment=3000_steps.yaml model.max_length=512 data.max_len=512

# python src/eval.py trainer=gpu logger=wandb ckpt_path=logs/train/runs/2024-04-18_20-49-13/checkpoints/epoch_000.ckpt data.batch_size=128
# python src/eval.py trainer=gpu logger=wandb ckpt_path=logs/train/runs/2024-04-16_11-08-47/checkpoints/epoch_000.ckpt data.batch_size=128 model.max_length=128 data.max_len=128
# python src/eval.py trainer=gpu logger=wandb ckpt_path=logs/train/runs/2024-04-16_21-35-02/checkpoints/epoch_000.ckpt data.batch_size=128 model.max_length=256 data.max_len=256
# python src/eval.py trainer=gpu logger=wandb ckpt_path=logs/train/runs/2024-04-17_09-45-16/checkpoints/epoch_000.ckpt data.batch_size=128 model.max_length=512 data.max_len=512

python src/train.py trainer=ddp logger=wandb experiment=3000_steps.yaml trainer.accumulate_grad_batches=16
python src/train.py trainer=ddp logger=wandb experiment=3000_steps.yaml trainer.accumulate_grad_batches=32
