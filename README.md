<div align="center">

# Harakaat

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

Automatic Arabic Text Diacritization <br>
[💥 Demo](https://huggingface.co/spaces/glonor/arabic-text-diacritization)

</div>

## About The Project

This project aims to develop a tool for Arabic text diacritization, which involves restoring the appropriate short-vowel markings (diacritics) to Arabic text. Diacritization is crucial for facilitating pronunciation and disambiguating words, as well as various natural language processing tasks such as text-to-speech synthesis.

## Features

- Based on Google's ByT5-small transformer model
- Fine-tuned on Taskheela Dataset

## Project Structure

The directory structure of new project looks like this:

```
├── configs                   <- Hydra configs
│
├── data                   <- Project data
│
├── logs                   <- Logs generated by hydra and lightning loggers
│
├── notebooks              <- Jupyter notebooks
│
├── scripts                <- Shell scripts
│
├── src                    <- Source code
│   ├── data                     <- Data scripts
│   ├── models                   <- Model scripts
│   ├── utils                    <- Utility scripts
│   │
│   ├── eval.py                  <- Run evaluation
│   └── train.py                 <- Run training
│
├── tests                  <- Tests of any kind
│
├── .env.example              <- Example of file for storing private environment variables
├── .gitignore                <- List of files ignored by git
├── .pre-commit-config.yaml   <- Configuration of pre-commit hooks for code formatting
├── .project-root             <- File for inferring the position of project root directory
├── environment.yaml          <- File for installing conda environment
├── Makefile                  <- Makefile with commands like `make train` or `make test`
├── pyproject.toml            <- Configuration options for testing and linting
├── setup.py                  <- File for installing project as a package
├── Dockerfile
└── README.md
```

<br>

## 🚀 Quickstart

#### Conda

```bash
# clone project
git clone https://github.com/glonor/harakaat
cd harakaat

# create conda environment and install dependencies
conda env create -f environment.yaml

# activate conda environment
conda activate har
```

## Run

```bash
python app/app.py
```

## How to train

Train model with default configuration

```bash
# train on GPU
python src/train.py trainer=gpu

# train on 2 GPUs
python src/train.py trainer=ddp
```

## How to evaluate

Evaluate checkpoint

```bash
python src/eval.py ckpt_path="/path/to/ckpt/name.ckpt"
```

## Dockerfile

This Dockerfile is for GPU only.

You will need to [install Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) to enable GPU support. <br>

To build the container use:

```bash
docker build -t harakaat .
```

To mount the project to the container use:

```bash
docker run -v $(pwd):/workspace/project --gpus all -it --rm harakaat
```

## Acknowledgments

- [Fine-Tashkeel: Finetuning Byte-Level Models for Accurate Arabic Text Diacritization (arxiv.org)](https://arxiv.org/abs/2303.14588)
- [Correcting diacritics and typos with a ByT5 transformer model (arxiv.org)](https://arxiv.org/abs/2201.13242)
- [google/byt5-small · Hugging Face](https://huggingface.co/google/byt5-small)
- [arbml/tashkeela · Dataset at Hugging Face](https://huggingface.co/datasets/arbml/tashkeela)
