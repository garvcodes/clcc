"""
Training script for multimodal contrastive learning.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path

from model import MultimodalEncoder, build_cnv_anchor_bank
from losses import CombinedLoss


def train_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    z_cnv_anchors,
    device,
    clip_grad_norm=1.0
):
    """
    Train for one epoch.
    
    Args:
        model: MultimodalEncoder
        dataloader: DataLoader with expression-CNV pairs
        criterion: CombinedLoss
        optimizer: torch optimizer
        z_cnv_anchors: (M, latent_dim) bank of CNV anchors
        device: torch device
        clip_grad_norm: max gradient norm for clipping
        
    Returns:
        avg_loss_dict: dict with average losses
    """
    model.train()
    
    # Reset centroids at start of epoch
    criterion.reset_centroids()
    
    epoch_losses = {
        'total': [],
        'contrastive': [],
        'centroid': [],
        'h_align': []
    }
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        x_expr = batch['x_expr'].to(device)
        x_cnv = batch['x_cnv'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        outputs = model(x_expr, x_cnv)
        
        # Compute loss
        loss, loss_dict = criterion(outputs, z_cnv_anchors, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        
        optimizer.step()
        
        # Track losses
        for key in epoch_losses:
            epoch_losses[key].append(loss_dict[key])
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss_dict['total']:.4f}",
            'contr': f"{loss_dict['contrastive']:.4f}",
            'cent': f"{loss_dict['centroid']:.4f}",
            'h': f"{loss_dict['h_align']:.4f}"
        })
    
    # Compute averages
    avg_loss_dict = {key: np.mean(vals) for key, vals in epoch_losses.items()}
    
    return avg_loss_dict


def train_model(
    model,
    dataset,
    cnv_profiles_tensor,
    n_epochs=100,
    batch_size=4096,
    learning_rate=1e-3,
    weight_decay=1e-4,
    device='cuda',
    save_dir='checkpoints',
    save_every=10
):
    """
    Full training loop.
    
    Args:
        model: MultimodalEncoder
        dataset: MultimodalScDataset
        cnv_profiles_tensor: (M, n_genes) tensor of CNV profiles
        n_epochs: number of training epochs
        batch_size: batch size
        learning_rate: learning rate
        weight_decay: weight decay for Adam
        device: torch device
        save_dir: where to save checkpoints
        save_every: save checkpoint every N epochs
        
    Returns:
        model: trained model
        history: dict with training history
    """
    # Setup
    model = model.to(device)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Loss and optimizer
    criterion = CombinedLoss(
        temperature=0.2,
        top_k_negatives=10,
        lambda_centroid=0.05,
        lambda_h_align=0.1
    )
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Training history
    history = {
        'total_loss': [],
        'contrastive_loss': [],
        'centroid_loss': [],
        'h_align_loss': []
    }
    
    print(f"\nStarting training for {n_epochs} epochs...")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Device: {device}\n")
    
    for epoch in range(1, n_epochs + 1):
        print(f"\nEpoch {epoch}/{n_epochs}")
        
        # Build CNV anchor bank (updated each epoch)
        z_cnv_anchors = build_cnv_anchor_bank(model, cnv_profiles_tensor, device)
        
        # Train for one epoch
        avg_losses = train_epoch(
            model, dataloader, criterion, optimizer,
            z_cnv_anchors, device
        )
        
        # Store history
        history['total_loss'].append(avg_losses['total'])
        history['contrastive_loss'].append(avg_losses['contrastive'])
        history['centroid_loss'].append(avg_losses['centroid'])
        history['h_align_loss'].append(avg_losses['h_align'])
        
        # Print epoch summary
        print(f"Epoch {epoch} - Avg Loss: {avg_losses['total']:.4f}")
        print(f"  Contrastive: {avg_losses['contrastive']:.4f}")
        print(f"  Centroid: {avg_losses['centroid']:.4f}")
        print(f"  H-align: {avg_losses['h_align']:.4f}")
        
        # Save checkpoint
        if epoch % save_every == 0 or epoch == n_epochs:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }
            torch.save(checkpoint, save_dir / f'checkpoint_epoch_{epoch}.pt')
            print(f"Saved checkpoint to {save_dir / f'checkpoint_epoch_{epoch}.pt'}")
    
    print("\nTraining complete!")
    return model, history


def load_checkpoint(model, checkpoint_path, device='cuda'):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    return model, checkpoint['history']


if __name__ == "__main__":
    # Example training script
    from data_processing import MultimodalScDataset
    import scanpy as sc
    
    print("This is an example training script.")
    print("To use it:")
    print("1. Load your preprocessed AnnData object")
    print("2. Load your CNV profiles")
    print("3. Create the dataset")
    print("4. Initialize the model")
    print("5. Call train_model()")
    print("\nSee notebooks/04_training.ipynb for a complete example.")
