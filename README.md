# Self-Supervised FGIR Pre-Training with Adaptive Sampling

This repository implements the method described in the paper **"Self-Supervised Pre-Training with Adaptive Sampling for Fine-Grained Image Retrieval"**. The project leverages self-supervised pre-training combined with an adaptive sampling strategy (ASS) to build a fine-grained image retrieval (FGIR) system.

## Main Features
- **Model Architecture:** Integrates contrastive and generative learning with an Adaptive Sample Selector (ASS) that dynamically selects training samples based on difficulty.
- **Experimental Setup:** Supports training and evaluation on datasets such as CUB-200, Cars-196, SOP, and In-Shop.
- **Code Structure:** Detailed in the repository structure below.

## Installation

We recommend creating a virtual environment using [conda](https://docs.conda.io/) or [virtualenv](https://virtualenv.pypa.io/):

```bash
pip install -r requirements.txt
```

## Training

Adjust the parameters in `experiments/config.yaml` as needed, then start training with:

```bash
python scripts/train.py --config experiments/config.yaml
```

## Evaluation

After training, evaluate your model using:

```
python scripts/evaluate.py --config experiments/config.yaml --checkpoint path/to/model_checkpoint.pth


```

## Repository Structure

- **src/**:Contains the model definitions, modules, dataset loader, and utility tools.
- **scripts/**: Contains training and evaluation scripts.
- **docs/**: Documentation for model architecture and usage.
- **tests/**: Unit tests for various modules.

## Citation

If you find this project helpful for your work, please cite our paper:

> Xiaoqing Li, Ya Wang, "Self-Supervised Pre-Training with Adaptive Sampling for Fine-Grained Image Retrieval", 2025.

