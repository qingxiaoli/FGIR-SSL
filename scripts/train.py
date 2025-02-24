#!/usr/bin/env python
import argparse
import yaml
import os
import torch
from torch.utils.data import DataLoader
from src.data.dataset import FGIRDataset
from src.model import SelfSupervisedModel
from src.modules.adaptive_sample_selector import AdaptiveSampleSelector
from src.modules.contrastive_loss import TripletContrastiveLoss, IntraImageContrastiveLoss
from src.modules.generative_loss import ReconstructionLoss
from src.utils.logger import get_logger

def train(config):
    logger = get_logger("TRAIN")
    device = torch.device(config['training'].get('device', 'cpu'))
    
    # Load dataset
    dataset = FGIRDataset(config['data']['data_dir'])
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'],
                            shuffle=True, num_workers=config['data'].get('num_workers', 4))
    
    # Build the model
    model = SelfSupervisedModel(backbone=config['model'].get('backbone', 'vit'),
                                pretrained=config['model'].get('pretrained', True))
    model = model.to(device)
    
    # Build the Adaptive Sample Selector (using the model's encoder as the feature extractor)
    ass = AdaptiveSampleSelector(model.encoder,
                                 sigma=config['training']['sigma'],
                                 Tmax=config['training']['Tmax'],
                                 device=device)
    
    # Define loss functions
    recon_loss_fn = ReconstructionLoss()
    triplet_loss_fn = TripletContrastiveLoss(margin=config['training']['margin'])
    intra_loss_fn = IntraImageContrastiveLoss(temperature=config['training']['temperature'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    for epoch in range(config['training']['epochs']):
        for batch in dataloader:
            images = batch['image'].to(device)
            # Here we use the same batch as candidate negatives; in practice, use a larger candidate pool
            negatives = images.clone()
            
            # Use ASS to select negative samples
            anchors, negative_samples = ass(images, negatives)
            
            # Forward pass
            shared_repr, reconstructed = model(images)
            
            # Generate a mask (here, a full mask of ones is used; implement random masking as needed)
            mask = torch.ones_like(images)
            loss_recon = recon_loss_fn(images, reconstructed, mask)
            
            loss_triplet = triplet_loss_fn(shared_repr, reconstructed, negative_samples)
            loss_intra = intra_loss_fn(shared_repr, reconstructed)
            
            # Total loss: L = lambda1 * L_recon + lambda2 * L_triplet + (1 - lambda2) * L_intra
            loss = config['training']['lambda1'] * loss_recon + \
                   config['training']['lambda2'] * loss_triplet + \
                   (1 - config['training']['lambda2']) * loss_intra
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        logger.info(f"Epoch {epoch+1}/{config['training']['epochs']} Loss: {loss.item():.4f}")
        # Optionally save a checkpoint after each epoch
        checkpoint_path = os.path.join("checkpoints", f"model_epoch{epoch+1}.pth")
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)

def main():
    parser = argparse.ArgumentParser(description="FGIR Self-Supervised Training")
    parser.add_argument('--config', type=str, required=True, help="Path to the config YAML file")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    train(config)

if __name__ == '__main__':
    main()
