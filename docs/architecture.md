# Model and Algorithm Architecture

This project is based on a self-supervised pre-training method that employs:

1. **Adaptive Sample Selector (ASS):**  
   Uses a lightweight feature extractor to encode input images and dynamically adjusts a difficulty factor (gamma) based on training progress to sample challenging negatives.  
   See `src/modules/adaptive_sample_selector.py`.

2. **Generative (Reconstruction) Module:**  
   Applies random masking and image reconstruction using Mean Squared Error (MSE) as the reconstruction loss.  
   See `src/modules/generative_loss.py`.

3. **Contrastive Learning Module:**  
   Contains both triplet contrastive loss and intra-image contrastive loss to pull positive samples closer and push negatives away.  
   See `src/modules/contrastive_loss.py`.

4. **Overall Loss Function:**  
   The total loss is defined as:  
   L = λ₁ * L_recon + λ₂ * L_triplet + (1 - λ₂) * L_intra

5. **Model Structure:**  
   The model uses a ViT-B/16 (or optionally ResNet) as the encoder, combined with a simple decoder for reconstruction.  
   See `src/model.py`.

This method effectively captures subtle differences in fine-grained image retrieval tasks.