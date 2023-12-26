# Standardizing Test Suite for Optimizer Experiments in the Era of Transformers

## Overview

Welcome to the Standardizing Test Suite for Optimizer Experiments repository by the [UT Statistical Learning & AI Group](https://www.cs.utexas.edu/~qlearning/). This repository is designed to provide a comprehensive and standardized testing framework for evaluating optimization algorithms in the context of transformer models. As transformers continue to play a pivotal role in natural language processing and other AI applications, the need for robust and standardized testing becomes crucial.

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
4. [Usage](#usage)
5. [Benchmarks](#benchmarks)
6. [Contributing](#contributing)
7. [License](#license)
8. [Citations](#citations)
9. [Acknowledgements](#acknowledgements)

## Introduction

In the rapidly evolving field of AI, transformer models have become indispensable tools for various tasks. However, optimizing these models poses unique challenges, and there is a growing need for a standardized approach to evaluate and compare different optimization algorithms. This repository addresses this need by providing a comprehensive test suite specifically tailored for transformer models.

## Features

- **Standardized Benchmarking:** Evaluate the performance of optimization algorithms using a standardized set of transformer models and datasets.
- **Reproducibility:** Ensure the reproducibility of experiments by providing configuration files and pre-processing scripts.
- **Scalability:** Test the scalability of optimization algorithms on models of different sizes and complexities.
- **Comprehensive Metrics:** Measure the performance of optimizers using a range of metrics relevant to transformer models.
- **Easy Integration:** Integrate the test suite seamlessly into your existing experimental framework.

## Getting Started

### Prerequisites

Ensure you have the following prerequisites installed:

- Python 3.x
- [PyTorch](https://pytorch.org/get-started/locally/)
- Other dependencies (specified in `requirements.txt`)

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/kyleliang919/Optimizer-Zoo.git
    cd Optimizer-Zoo
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

Example usage:

1. Run the pretraining experiment on gpt2 and openwebtext:

    ```bash
    torchrun --nproc_per_node 4 -m run_clm \
    --config_name gpt2 \
    --tokenizer_name gpt2 \
    --dataset_name openwebtext \
    --per_device_train_batch_size 20 \
    --per_device_eval_batch_size 24 \
    --do_train \
    --do_eval \
    --output_dir result/gpt2_lion_wd_0.1 \
    --report_to wandb \
    --torch_dtype bfloat16 \
    --gradient_accumulation_steps 8 \
    --max_steps 100000 \
    --warmup_steps 2000 \
    --optim lion \
    --save_total_limit 2 \
    --learning_rate 0.0001 \
    --weight_decay 0.1 \
    --async_grad
    ```

2. Run the SFT experiment on llama7B and stack-exchanged-pair
    ```
    torchrun --nproc_per_node 4 -m sft_llama2 \
    --output_dir="./sft" \
    --max_steps=500 \
    --logging_steps=10 \
    --save_steps=10 \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps=2 \
    --gradient_checkpointing=False \
    --group_by_length=False \
    --learning_rate=1e-4 \
    --lr_scheduler_type="cosine" \
    --warmup_steps=100 \
    --weight_decay=0.05 \
    --optim="paged_adamw_32bit" \
    --bf16=True \
    --remove_unused_columns=False \
    --run_name="sft_llama2" \
    --report_to="wandb" \
    --optim lion \
     --async_grad
    ```

    

3. View results and analysis in the `results` directory.

For detailed usage instructions, refer to the [user documentation](docs/user-docs.md).

## Benchmarks
Please refer to the following tables for the performance of different optimizers

## Contributing

We welcome contributions from the community. If you would like to contribute, please follow our [contribution guidelines](CONTRIBUTING.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citations

If you use this test suite in your research or work, please consider citing the following relevant publications:

1. [Chen, Lizhang, et al. "Lion Secretly Solves Constrained Optimization: As Lyapunov Predicts." arXiv preprint arXiv:2310.05898 (2023).](https://arxiv.org/abs/2310.05898) [[Bibtex]](https://scholar.googleusercontent.com/scholar.bib?q=info:VSH2VlwlnCoJ:scholar.google.com/&output=citation&scisdr=ClH2GPDxEMyG8inDrmE:AFWwaeYAAAAAZYXFtmFYCW7Y8CwiBWAoZU665a8&scisig=AFWwaeYAAAAAZYXFtj5e_waaw90lqnrPpQVHSy8&scisf=4&ct=citation&cd=-1&hl=en)

## Acknowledgements

We would like to express our gratitude to the contributors and researchers who have made this project possible. Special thanks to [list of contributors] for their valuable input and feedback.

For more information about the UT Statistical Learning & AI Group, visit [our website](https://www.utstat.ai).

---

Feel free to customize this README to suit the specific details of your repository and provide more specific information about the transformer models and datasets used in your test suite.
