"""
Contrastive learning model for aligning scRNA-seq expression with CNV profiles.

This script trains a model that learns to embed expression and CNV data into
a shared latent space where matching pairs (same cell) are close together.

Usage:
    python 07_contrastive_model.py --train
    python 07_contrastive_model.py --evaluate
"""

import os
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Dataset
# ============================================================================

class ExpressionCNVDataset(Dataset):
    """Dataset that pairs expression vectors with CNV profiles for the same cell."""

    def __init__(
        self,
        expression_data: np.ndarray,
        cnv_data: np.ndarray,
        cell_ids: np.ndarray,
        metadata: Optional[pd.DataFrame] = None
    ):
        """
        Args:
            expression_data: (n_cells, n_genes) expression matrix
            cnv_data: (n_cells, n_cnv_windows) CNV matrix
            cell_ids: Cell identifiers
            metadata: Optional cell metadata
        """
        assert len(expression_data) == len(cnv_data), "Expression and CNV must have same number of cells"

        self.expression = torch.FloatTensor(expression_data)
        self.cnv = torch.FloatTensor(cnv_data)
        self.cell_ids = cell_ids
        self.metadata = metadata

    def __len__(self):
        return len(self.expression)

    def __getitem__(self, idx):
        return {
            'expression': self.expression[idx],
            'cnv': self.cnv[idx],
            'idx': idx
        }


# ============================================================================
# Model Architecture
# ============================================================================

class ExpressionEncoder(nn.Module):
    """Encoder for gene expression data."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256],
        latent_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, latent_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return F.normalize(self.encoder(x), dim=-1)


class CNVEncoder(nn.Module):
    """Encoder for CNV profile data."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128],
        latent_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, latent_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return F.normalize(self.encoder(x), dim=-1)


class ContrastiveModel(nn.Module):
    """
    Contrastive learning model that aligns expression and CNV embeddings.

    Uses InfoNCE loss to learn a shared embedding space where matching
    expression-CNV pairs are close together.
    """

    def __init__(
        self,
        expression_dim: int,
        cnv_dim: int,
        latent_dim: int = 128,
        expression_hidden: List[int] = [512, 256],
        cnv_hidden: List[int] = [256, 128],
        temperature: float = 0.07,
        dropout: float = 0.1
    ):
        super().__init__()

        self.expression_encoder = ExpressionEncoder(
            input_dim=expression_dim,
            hidden_dims=expression_hidden,
            latent_dim=latent_dim,
            dropout=dropout
        )

        self.cnv_encoder = CNVEncoder(
            input_dim=cnv_dim,
            hidden_dims=cnv_hidden,
            latent_dim=latent_dim,
            dropout=dropout
        )

        self.temperature = temperature
        self.latent_dim = latent_dim

    def forward(self, expression, cnv):
        """
        Forward pass returning embeddings for both modalities.

        Args:
            expression: (batch_size, expression_dim) expression vectors
            cnv: (batch_size, cnv_dim) CNV vectors

        Returns:
            expression_embed: (batch_size, latent_dim) normalized embeddings
            cnv_embed: (batch_size, latent_dim) normalized embeddings
        """
        expression_embed = self.expression_encoder(expression)
        cnv_embed = self.cnv_encoder(cnv)
        return expression_embed, cnv_embed

    def compute_loss(self, expression_embed, cnv_embed):
        """
        Compute symmetric InfoNCE contrastive loss.

        Matching pairs (same cell) should have high similarity,
        non-matching pairs should have low similarity.
        """
        batch_size = expression_embed.shape[0]

        # Compute similarity matrix
        logits = torch.matmul(expression_embed, cnv_embed.T) / self.temperature

        # Labels: diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=logits.device)

        # Symmetric loss: expression->CNV and CNV->expression
        loss_exp_to_cnv = F.cross_entropy(logits, labels)
        loss_cnv_to_exp = F.cross_entropy(logits.T, labels)

        loss = (loss_exp_to_cnv + loss_cnv_to_exp) / 2

        # Compute accuracy (how often the correct pair has highest similarity)
        with torch.no_grad():
            pred_exp = logits.argmax(dim=1)
            pred_cnv = logits.T.argmax(dim=1)
            acc_exp = (pred_exp == labels).float().mean()
            acc_cnv = (pred_cnv == labels).float().mean()
            accuracy = (acc_exp + acc_cnv) / 2

        return loss, accuracy


# ============================================================================
# Data Loading
# ============================================================================

def load_patient_data(
    patient_id: str,
    processed_dir: str = 'data/processed',
    cnv_dir: str = 'data/cnv_output'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load expression and CNV data for a patient.

    Returns:
        expression: (n_cells, n_genes) matrix
        cnv: (n_cells, n_cnv_windows) matrix
        cell_ids: Cell identifiers
        metadata: Cell metadata
    """
    # Load CNV data (contains both expression and CNV)
    cnv_path = os.path.join(cnv_dir, patient_id, f'{patient_id}_cnv.h5ad')
    if not os.path.exists(cnv_path):
        raise FileNotFoundError(f"CNV file not found: {cnv_path}")

    print(f"Loading {patient_id}...")
    adata = sc.read_h5ad(cnv_path)

    # Get expression matrix (stored in .X after CNV processing)
    if sparse.issparse(adata.X):
        expression = adata.X.toarray()
    else:
        expression = np.array(adata.X)

    # Get CNV matrix
    cnv = adata.obsm['X_cnv']
    if sparse.issparse(cnv):
        cnv = cnv.toarray()
    else:
        cnv = np.array(cnv)

    # Get cell IDs and metadata
    cell_ids = adata.obs_names.values
    metadata = adata.obs.copy()

    print(f"  Cells: {len(cell_ids):,}")
    print(f"  Expression dim: {expression.shape[1]:,}")
    print(f"  CNV dim: {cnv.shape[1]:,}")

    return expression, cnv, cell_ids, metadata


def load_all_patients(
    patient_ids: List[str],
    processed_dir: str = 'data/processed',
    cnv_dir: str = 'data/cnv_output'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Load and concatenate data from multiple patients."""

    all_expression = []
    all_cnv = []
    all_cell_ids = []
    all_metadata = []

    for patient_id in patient_ids:
        try:
            expr, cnv, cells, meta = load_patient_data(
                patient_id, processed_dir, cnv_dir
            )

            # Add patient ID to cell names to ensure uniqueness
            cells = np.array([f"{patient_id}_{c}" for c in cells])
            meta['patient_id'] = patient_id

            all_expression.append(expr)
            all_cnv.append(cnv)
            all_cell_ids.append(cells)
            all_metadata.append(meta)

        except Exception as e:
            print(f"Warning: Could not load {patient_id}: {e}")
            continue

    if not all_expression:
        raise ValueError("No patient data could be loaded")

    # Concatenate all data
    expression = np.vstack(all_expression)
    cnv = np.vstack(all_cnv)
    cell_ids = np.concatenate(all_cell_ids)
    metadata = pd.concat(all_metadata, ignore_index=False)

    print(f"\nTotal: {len(cell_ids):,} cells from {len(all_expression)} patients")

    return expression, cnv, cell_ids, metadata


def prepare_data(
    patient_ids: List[str],
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    random_seed: int = 42
) -> Dict[str, ExpressionCNVDataset]:
    """
    Prepare train/val/test datasets.

    Args:
        patient_ids: List of patient IDs to include
        val_fraction: Fraction of data for validation
        test_fraction: Fraction of data for testing
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with 'train', 'val', 'test' datasets
    """
    np.random.seed(random_seed)

    # Load all data
    expression, cnv, cell_ids, metadata = load_all_patients(patient_ids)

    # Normalize expression data (z-score per gene)
    expression_mean = expression.mean(axis=0)
    expression_std = expression.std(axis=0) + 1e-8
    expression = (expression - expression_mean) / expression_std

    # Normalize CNV data
    cnv_mean = cnv.mean(axis=0)
    cnv_std = cnv.std(axis=0) + 1e-8
    cnv = (cnv - cnv_mean) / cnv_std

    # Split indices
    n_cells = len(cell_ids)
    indices = np.random.permutation(n_cells)

    n_test = int(n_cells * test_fraction)
    n_val = int(n_cells * val_fraction)

    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    print(f"\nSplit: {len(train_idx):,} train, {len(val_idx):,} val, {len(test_idx):,} test")

    # Create datasets
    datasets = {
        'train': ExpressionCNVDataset(
            expression[train_idx],
            cnv[train_idx],
            cell_ids[train_idx],
            metadata.iloc[train_idx]
        ),
        'val': ExpressionCNVDataset(
            expression[val_idx],
            cnv[val_idx],
            cell_ids[val_idx],
            metadata.iloc[val_idx]
        ),
        'test': ExpressionCNVDataset(
            expression[test_idx],
            cnv[test_idx],
            cell_ids[test_idx],
            metadata.iloc[test_idx]
        )
    }

    # Store normalization parameters for inference
    datasets['normalization'] = {
        'expression_mean': expression_mean,
        'expression_std': expression_std,
        'cnv_mean': cnv_mean,
        'cnv_std': cnv_std
    }

    return datasets


# ============================================================================
# Training
# ============================================================================

def train_epoch(
    model: ContrastiveModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_acc = 0
    n_batches = 0

    for batch in dataloader:
        expression = batch['expression'].to(device)
        cnv = batch['cnv'].to(device)

        optimizer.zero_grad()

        exp_embed, cnv_embed = model(expression, cnv)
        loss, acc = model.compute_loss(exp_embed, cnv_embed)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += acc.item()
        n_batches += 1

    return total_loss / n_batches, total_acc / n_batches


def evaluate(
    model: ContrastiveModel,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[float, float]:
    """Evaluate model."""
    model.eval()
    total_loss = 0
    total_acc = 0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            expression = batch['expression'].to(device)
            cnv = batch['cnv'].to(device)

            exp_embed, cnv_embed = model(expression, cnv)
            loss, acc = model.compute_loss(exp_embed, cnv_embed)

            total_loss += loss.item()
            total_acc += acc.item()
            n_batches += 1

    return total_loss / n_batches, total_acc / n_batches


def train_model(
    datasets: Dict[str, ExpressionCNVDataset],
    output_dir: str = 'models/contrastive',
    latent_dim: int = 128,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    n_epochs: int = 100,
    patience: int = 10,
    device: str = 'auto'
) -> ContrastiveModel:
    """
    Train the contrastive model.

    Args:
        datasets: Dictionary with train/val/test datasets
        output_dir: Where to save model checkpoints
        latent_dim: Dimension of shared latent space
        batch_size: Training batch size
        learning_rate: Learning rate
        weight_decay: L2 regularization
        n_epochs: Maximum number of epochs
        patience: Early stopping patience
        device: 'cuda', 'mps', 'cpu', or 'auto'

    Returns:
        Trained model
    """
    os.makedirs(output_dir, exist_ok=True)

    # Select device
    if device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device)
    print(f"\nUsing device: {device}")

    # Get dimensions
    expression_dim = datasets['train'].expression.shape[1]
    cnv_dim = datasets['train'].cnv.shape[1]

    print(f"Expression dimension: {expression_dim}")
    print(f"CNV dimension: {cnv_dim}")
    print(f"Latent dimension: {latent_dim}")

    # Create model
    model = ContrastiveModel(
        expression_dim=expression_dim,
        cnv_dim=cnv_dim,
        latent_dim=latent_dim
    ).to(device)

    # Create data loaders
    train_loader = DataLoader(
        datasets['train'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    val_loader = DataLoader(
        datasets['val'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=n_epochs
    )

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"\nTraining for up to {n_epochs} epochs...")
    print("-" * 60)

    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1:3d} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'expression_dim': expression_dim,
                'cnv_dim': cnv_dim,
                'latent_dim': latent_dim,
                'normalization': datasets['normalization']
            }, os.path.join(output_dir, 'best_model.pt'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)

    # Load best model
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"\nBest model from epoch {checkpoint['epoch']+1}")
    print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    print(f"  Val Accuracy: {checkpoint['val_acc']:.4f}")

    return model


# ============================================================================
# Evaluation
# ============================================================================

def extract_embeddings(
    model: ContrastiveModel,
    dataset: ExpressionCNVDataset,
    device: torch.device,
    batch_size: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract embeddings for all cells in dataset."""
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    exp_embeds = []
    cnv_embeds = []

    with torch.no_grad():
        for batch in loader:
            expression = batch['expression'].to(device)
            cnv = batch['cnv'].to(device)

            exp_embed, cnv_embed = model(expression, cnv)

            exp_embeds.append(exp_embed.cpu().numpy())
            cnv_embeds.append(cnv_embed.cpu().numpy())

    return np.vstack(exp_embeds), np.vstack(cnv_embeds)


def visualize_embedding_space(
    model: ContrastiveModel,
    dataset: ExpressionCNVDataset,
    device: torch.device,
    output_dir: str,
    method: str = 'umap',
    sample_size: Optional[int] = None
):
    """
    Visualize the entire embedding space after training.

    Creates visualizations showing:
    1. Expression and CNV embeddings in shared space (colored by modality)
    2. Embeddings colored by cancer vs normal status
    3. Embeddings colored by CNV subcluster
    4. Embedding distance distribution
    5. Alignment quality (lines connecting matched pairs)

    Args:
        model: Trained contrastive model
        dataset: Dataset to visualize
        device: Torch device
        output_dir: Directory to save plots
        method: Dimensionality reduction method ('umap' or 'pca')
        sample_size: Optional limit on number of cells to visualize
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    try:
        import umap
        UMAP_AVAILABLE = True
    except ImportError:
        UMAP_AVAILABLE = False
        if method == 'umap':
            print("UMAP not available, falling back to PCA")
            method = 'pca'

    os.makedirs(output_dir, exist_ok=True)

    print("\nExtracting embeddings for visualization...")
    exp_embeds, cnv_embeds = extract_embeddings(model, dataset, device)

    # Get metadata
    metadata = dataset.metadata

    # Optionally subsample for faster visualization
    n_cells = len(exp_embeds)
    if sample_size and sample_size < n_cells:
        np.random.seed(42)
        idx = np.random.choice(n_cells, sample_size, replace=False)
        exp_embeds = exp_embeds[idx]
        cnv_embeds = cnv_embeds[idx]
        metadata = metadata.iloc[idx]
        print(f"Subsampled to {sample_size} cells for visualization")

    n_viz = len(exp_embeds)

    # Combine embeddings for joint dimensionality reduction
    combined = np.vstack([exp_embeds, cnv_embeds])
    modality_labels = ['Expression'] * n_viz + ['CNV'] * n_viz

    # Dimensionality reduction
    print(f"Running {method.upper()} dimensionality reduction...")
    if method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3)
        coords_2d = reducer.fit_transform(combined)
    else:
        reducer = PCA(n_components=2, random_state=42)
        coords_2d = reducer.fit_transform(combined)

    exp_coords = coords_2d[:n_viz]
    cnv_coords = coords_2d[n_viz:]

    # Calculate embedding distances (1 - cosine similarity)
    similarities = (exp_embeds * cnv_embeds).sum(axis=1)
    distances = 1 - similarities

    # -------------------------------------------------------------------------
    # Plot 1: Expression vs CNV embeddings (by modality)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(exp_coords[:, 0], exp_coords[:, 1], c='blue', alpha=0.5, s=10, label='Expression')
    ax.scatter(cnv_coords[:, 0], cnv_coords[:, 1], c='red', alpha=0.5, s=10, label='CNV')

    ax.set_xlabel(f'{method.upper()}1')
    ax.set_ylabel(f'{method.upper()}2')
    ax.set_title('Embedding Space: Expression vs CNV Modalities')
    ax.legend(markerscale=3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'embedding_by_modality.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/embedding_by_modality.png")

    # -------------------------------------------------------------------------
    # Plot 2: Embeddings colored by cancer vs normal
    # -------------------------------------------------------------------------
    if 'cancer_vs_normal' in metadata.columns:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        cancer_mask = metadata['cancer_vs_normal'].values == 'Cancer'

        # Expression embeddings
        for label, mask, color in [('Normal', ~cancer_mask, 'blue'), ('Cancer', cancer_mask, 'red')]:
            if mask.sum() > 0:
                axes[0].scatter(exp_coords[mask, 0], exp_coords[mask, 1],
                               c=color, alpha=0.5, s=10, label=label)
        axes[0].set_xlabel(f'{method.upper()}1')
        axes[0].set_ylabel(f'{method.upper()}2')
        axes[0].set_title('Expression Embeddings: Cancer vs Normal')
        axes[0].legend(markerscale=3)

        # CNV embeddings
        for label, mask, color in [('Normal', ~cancer_mask, 'blue'), ('Cancer', cancer_mask, 'red')]:
            if mask.sum() > 0:
                axes[1].scatter(cnv_coords[mask, 0], cnv_coords[mask, 1],
                               c=color, alpha=0.5, s=10, label=label)
        axes[1].set_xlabel(f'{method.upper()}1')
        axes[1].set_ylabel(f'{method.upper()}2')
        axes[1].set_title('CNV Embeddings: Cancer vs Normal')
        axes[1].legend(markerscale=3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'embedding_cancer_vs_normal.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir}/embedding_cancer_vs_normal.png")

    # -------------------------------------------------------------------------
    # Plot 3: Embeddings colored by CNV subcluster
    # -------------------------------------------------------------------------
    if 'cnv_leiden' in metadata.columns:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        subclusters = metadata['cnv_leiden'].values
        n_clusters = len(np.unique(subclusters))
        cmap = plt.cm.get_cmap('tab20', n_clusters)

        sc1 = axes[0].scatter(exp_coords[:, 0], exp_coords[:, 1],
                              c=subclusters.astype(int), cmap=cmap, alpha=0.5, s=10)
        axes[0].set_xlabel(f'{method.upper()}1')
        axes[0].set_ylabel(f'{method.upper()}2')
        axes[0].set_title('Expression Embeddings: CNV Subclusters')
        plt.colorbar(sc1, ax=axes[0], label='Subcluster')

        sc2 = axes[1].scatter(cnv_coords[:, 0], cnv_coords[:, 1],
                              c=subclusters.astype(int), cmap=cmap, alpha=0.5, s=10)
        axes[1].set_xlabel(f'{method.upper()}1')
        axes[1].set_ylabel(f'{method.upper()}2')
        axes[1].set_title('CNV Embeddings: CNV Subclusters')
        plt.colorbar(sc2, ax=axes[1], label='Subcluster')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'embedding_by_subcluster.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir}/embedding_by_subcluster.png")

    # -------------------------------------------------------------------------
    # Plot 4: Embedding distance distribution
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of distances
    axes[0].hist(distances, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(np.median(distances), color='red', linestyle='--',
                    label=f'Median: {np.median(distances):.3f}')
    axes[0].set_xlabel('Embedding Distance (1 - cosine similarity)')
    axes[0].set_ylabel('Number of Cells')
    axes[0].set_title('Expression-CNV Embedding Distance Distribution')
    axes[0].legend()

    # Distance by cancer vs normal
    if 'cancer_vs_normal' in metadata.columns:
        cancer_mask = metadata['cancer_vs_normal'].values == 'Cancer'
        axes[1].hist(distances[~cancer_mask], bins=30, alpha=0.6, label='Normal', color='blue')
        axes[1].hist(distances[cancer_mask], bins=30, alpha=0.6, label='Cancer', color='red')
        axes[1].set_xlabel('Embedding Distance')
        axes[1].set_ylabel('Number of Cells')
        axes[1].set_title('Embedding Distance: Cancer vs Normal')
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'embedding_distance_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/embedding_distance_distribution.png")

    # -------------------------------------------------------------------------
    # Plot 5: Alignment visualization (connecting matched pairs)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot points
    ax.scatter(exp_coords[:, 0], exp_coords[:, 1], c='blue', alpha=0.3, s=15, label='Expression')
    ax.scatter(cnv_coords[:, 0], cnv_coords[:, 1], c='red', alpha=0.3, s=15, label='CNV')

    # Draw lines connecting matched pairs (subsample for clarity)
    n_lines = min(200, n_viz)
    line_idx = np.random.choice(n_viz, n_lines, replace=False)

    for i in line_idx:
        # Color line by distance (green=close, red=far)
        dist = distances[i]
        color = plt.cm.RdYlGn_r(dist / distances.max())
        ax.plot([exp_coords[i, 0], cnv_coords[i, 0]],
                [exp_coords[i, 1], cnv_coords[i, 1]],
                c=color, alpha=0.3, linewidth=0.5)

    ax.set_xlabel(f'{method.upper()}1')
    ax.set_ylabel(f'{method.upper()}2')
    ax.set_title('Embedding Alignment\n(Lines connect expression-CNV pairs for same cell)')
    ax.legend(markerscale=3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'embedding_alignment.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/embedding_alignment.png")

    # -------------------------------------------------------------------------
    # Plot 6: Combined view with distance coloring
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Expression embeddings colored by distance
    sc1 = axes[0].scatter(exp_coords[:, 0], exp_coords[:, 1],
                          c=distances, cmap='RdYlGn_r', alpha=0.6, s=15)
    axes[0].set_xlabel(f'{method.upper()}1')
    axes[0].set_ylabel(f'{method.upper()}2')
    axes[0].set_title('Expression Embeddings\n(Color = distance to matched CNV)')
    plt.colorbar(sc1, ax=axes[0], label='Embedding Distance')

    # CNV embeddings colored by distance
    sc2 = axes[1].scatter(cnv_coords[:, 0], cnv_coords[:, 1],
                          c=distances, cmap='RdYlGn_r', alpha=0.6, s=15)
    axes[1].set_xlabel(f'{method.upper()}1')
    axes[1].set_ylabel(f'{method.upper()}2')
    axes[1].set_title('CNV Embeddings\n(Color = distance to matched expression)')
    plt.colorbar(sc2, ax=axes[1], label='Embedding Distance')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'embedding_by_distance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/embedding_by_distance.png")

    # -------------------------------------------------------------------------
    # Summary statistics
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Embedding Space Summary")
    print("=" * 50)
    print(f"Total cells visualized: {n_viz:,}")
    print(f"Embedding dimension: {exp_embeds.shape[1]}")
    print(f"\nEmbedding distances:")
    print(f"  Mean: {distances.mean():.4f}")
    print(f"  Median: {np.median(distances):.4f}")
    print(f"  Std: {distances.std():.4f}")
    print(f"  Min: {distances.min():.4f}")
    print(f"  Max: {distances.max():.4f}")

    if 'cancer_vs_normal' in metadata.columns:
        cancer_mask = metadata['cancer_vs_normal'].values == 'Cancer'
        print(f"\nBy cancer status:")
        print(f"  Normal mean distance: {distances[~cancer_mask].mean():.4f}")
        print(f"  Cancer mean distance: {distances[cancer_mask].mean():.4f}")

    # Save embedding data for further analysis
    embed_df = pd.DataFrame({
        'cell_id': dataset.cell_ids[:n_viz] if sample_size else dataset.cell_ids,
        'exp_umap1': exp_coords[:, 0],
        'exp_umap2': exp_coords[:, 1],
        'cnv_umap1': cnv_coords[:, 0],
        'cnv_umap2': cnv_coords[:, 1],
        'embedding_distance': distances
    })

    # Add metadata columns
    for col in ['cancer_vs_normal', 'cnv_leiden', 'cnv_score']:
        if col in metadata.columns:
            embed_df[col] = metadata[col].values

    embed_df.to_csv(os.path.join(output_dir, 'embedding_coordinates.csv'), index=False)
    print(f"\nSaved: {output_dir}/embedding_coordinates.csv")


def evaluate_retrieval(
    model: ContrastiveModel,
    dataset: ExpressionCNVDataset,
    device: torch.device,
    k_values: List[int] = [1, 5, 10, 50]
) -> Dict[str, float]:
    """
    Evaluate retrieval performance.

    For each expression embedding, find the k nearest CNV embeddings
    and check if the correct match is among them (and vice versa).
    """
    exp_embeds, cnv_embeds = extract_embeddings(model, dataset, device)

    # Compute similarity matrix
    similarity = np.matmul(exp_embeds, cnv_embeds.T)

    n_samples = len(similarity)
    results = {}

    # Expression -> CNV retrieval
    for k in k_values:
        top_k = np.argsort(-similarity, axis=1)[:, :k]
        correct = np.array([i in top_k[i] for i in range(n_samples)])
        results[f'exp_to_cnv_R@{k}'] = correct.mean()

    # CNV -> Expression retrieval
    for k in k_values:
        top_k = np.argsort(-similarity.T, axis=1)[:, :k]
        correct = np.array([i in top_k[i] for i in range(n_samples)])
        results[f'cnv_to_exp_R@{k}'] = correct.mean()

    return results


# ============================================================================
# Main
# ============================================================================

def get_available_patients(cnv_dir: str = 'data/cnv_output') -> List[str]:
    """Get list of patients with CNV data."""
    patients = []
    if os.path.exists(cnv_dir):
        for name in os.listdir(cnv_dir):
            if name.startswith('P') and os.path.isdir(os.path.join(cnv_dir, name)):
                cnv_file = os.path.join(cnv_dir, name, f'{name}_cnv.h5ad')
                if os.path.exists(cnv_file):
                    patients.append(name)
    return sorted(patients)


def main():
    parser = argparse.ArgumentParser(
        description='Train contrastive model for expression-CNV alignment'
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train the model'
    )
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Evaluate trained model'
    )
    parser.add_argument(
        '--patients',
        type=str,
        nargs='+',
        default=None,
        help='Patient IDs to use (default: all available)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/contrastive',
        help='Output directory for model'
    )
    parser.add_argument(
        '--latent-dim',
        type=int,
        default=128,
        help='Latent space dimension'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Batch size'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Maximum epochs'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Visualize the embedding space after training'
    )
    parser.add_argument(
        '--viz-method',
        type=str,
        default='umap',
        choices=['umap', 'pca'],
        help='Dimensionality reduction method for visualization'
    )
    parser.add_argument(
        '--viz-sample',
        type=int,
        default=None,
        help='Number of cells to sample for visualization (default: all)'
    )

    args = parser.parse_args()

    # Get patients
    if args.patients:
        patient_ids = args.patients
    else:
        patient_ids = get_available_patients()

    if not patient_ids:
        print("No patients with CNV data found!")
        print("Run: python 05_run_infercnv.py --all-patients")
        return

    print("=" * 60)
    print("Contrastive Learning: Expression-CNV Alignment")
    print("=" * 60)
    print(f"\nPatients: {patient_ids}")

    if args.train:
        # Prepare data
        print("\nPreparing data...")
        datasets = prepare_data(patient_ids)

        # Train model
        model = train_model(
            datasets,
            output_dir=args.output_dir,
            latent_dim=args.latent_dim,
            batch_size=args.batch_size,
            n_epochs=args.epochs,
            learning_rate=args.lr
        )

        # Evaluate on test set
        print("\n" + "=" * 60)
        print("Test Set Evaluation")
        print("=" * 60)

        device = next(model.parameters()).device
        test_loss, test_acc = evaluate(
            model,
            DataLoader(datasets['test'], batch_size=args.batch_size),
            device
        )
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")

        # Retrieval metrics
        retrieval = evaluate_retrieval(model, datasets['test'], device)
        print("\nRetrieval Performance:")
        for metric, value in retrieval.items():
            print(f"  {metric}: {value:.4f}")

        print(f"\nModel saved to: {args.output_dir}")

        # Visualize if requested
        if args.visualize:
            print("\n" + "=" * 60)
            print("Visualizing Embedding Space")
            print("=" * 60)
            visualize_embedding_space(
                model, datasets['test'], device,
                output_dir=args.output_dir,
                method=args.viz_method,
                sample_size=args.viz_sample
            )

    elif args.visualize:
        # Just visualize existing model
        model_path = os.path.join(args.output_dir, 'best_model.pt')
        if not os.path.exists(model_path):
            print(f"No trained model found at {model_path}")
            print("Run with --train first")
            return

        print(f"\nLoading model from {model_path}")
        checkpoint = torch.load(model_path, weights_only=False)

        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

        model = ContrastiveModel(
            expression_dim=checkpoint['expression_dim'],
            cnv_dim=checkpoint['cnv_dim'],
            latent_dim=checkpoint['latent_dim']
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

        datasets = prepare_data(patient_ids)

        print("\n" + "=" * 60)
        print("Visualizing Embedding Space")
        print("=" * 60)
        visualize_embedding_space(
            model, datasets['test'], device,
            output_dir=args.output_dir,
            method=args.viz_method,
            sample_size=args.viz_sample
        )

    elif args.evaluate:
        # Load model and evaluate
        model_path = os.path.join(args.output_dir, 'best_model.pt')
        if not os.path.exists(model_path):
            print(f"No trained model found at {model_path}")
            print("Run with --train first")
            return

        print(f"\nLoading model from {model_path}")
        checkpoint = torch.load(model_path, weights_only=False)

        # Determine device
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

        model = ContrastiveModel(
            expression_dim=checkpoint['expression_dim'],
            cnv_dim=checkpoint['cnv_dim'],
            latent_dim=checkpoint['latent_dim']
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Prepare test data
        datasets = prepare_data(patient_ids)

        # Evaluate
        test_loss, test_acc = evaluate(
            model,
            DataLoader(datasets['test'], batch_size=args.batch_size),
            device
        )
        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")

        retrieval = evaluate_retrieval(model, datasets['test'], device)
        print("\nRetrieval Performance:")
        for metric, value in retrieval.items():
            print(f"  {metric}: {value:.4f}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
