# Usage Instructions

## Data Preparation

Place your fine-grained image dataset (e.g., CUB-200, Cars-196) in the directory `./data/FGIR_dataset`. 

## Modify Configuration

Edit `experiments/config.yaml` to adjust the data path, training parameters, and model settings.

## Training the Model

Start training by running:

```bash
python scripts/train.py --config experiments/config.yaml
```

## Model Evaluation

Once training is complete, evaluate the model using:

```
python scripts/evaluate.py --config experiments/config.yaml --checkpoint path/to/model_checkpoint.pth
```

Evaluation metrics (e.g., Recall@K) will be printed on the console.

