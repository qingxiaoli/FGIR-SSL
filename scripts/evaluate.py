#!/usr/bin/env python
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from src.data.dataset import FGIRDataset
from src.model import SelfSupervisedModel
from src.utils.metrics import compute_recall
from src.utils.logger import get_logger

def evaluate(config, checkpoint_path):
    logger = get_logger("EVALUATE")
    device = torch.device(config['training'].get('device', 'cpu'))
    
    # Load dataset
    dataset = FGIRDataset(config['data']['data_dir'])
    dataloader = DataLoader(dataset, batch_size=config['evaluation']['batch_size'],
                            shuffle=False, num_workers=config['data'].get('num_workers', 4))
    
    # Build the model
    model = SelfSupervisedModel(backbone=config['model'].get('backbone', 'vit'),
                                pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Extract features
    features = []
    labels = []  # If your dataset provides labels, collect them here
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            feats, _ = model(images)
            features.append(feats.cpu())
            # If labels are provided, append: labels.extend(batch['label'])
    
    features = torch.cat(features, dim=0)
    # Example: compute recall using Euclidean distance; modify as needed for your task
    recall = compute_recall(features, features, k_list=config['evaluation']['metrics'])
    logger.info("Evaluation Metrics:")
    for k, v in recall.items():
        logger.info(f"{k}: {v:.2f}%")

def main():
    parser = argparse.ArgumentParser(description="FGIR Model Evaluation")
    parser.add_argument('--config', type=str, required=True, help="Path to the config YAML file")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the model checkpoint")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    evaluate(config, args.checkpoint)

if __name__ == '__main__':
    main()
