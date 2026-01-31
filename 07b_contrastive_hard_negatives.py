"""
Contrastive learning with HARD NEGATIVE MINING for better gene-level signal.

The original contrastive model (07_contrastive_model.py) learns to align
expression and CNV embeddings for matched pairs. However, this doesn't
explicitly learn what makes concordant vs discordant cells different.

This version adds hard negative mining:
1. Standard positive pairs: (expression_i, cnv_i) for same cell
2. Hard negatives: cells with similar CNV but different expression patterns
   - These are cells that SHOULD have similar expression (same CNV) but DON'T
   - Pushing these apart teaches the model to recognize discordance

The key insight: if two cells have the same CNV but different expression,
at least one of them is "discordant" - its expression doesn't follow CNV.
By explicitly contrasting these, we learn a space where discordance is meaningful.

Usage:
    python 07b_contrastive_hard_negatives.py --train
    python 07b_contrastive_hard_negatives.py --train --hard-negative-weight 0.5
"""

import os
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.spatial.distance import cdist
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Dataset with Hard Negative Mining
# ============================================================================

class HardNegativeDataset(Dataset):
    """
    Dataset that provides:
    1. Positive pairs: (expression_i, cnv_i) for same cell
    2. Hard negatives: cells with similar CNV but different expression
    """

    def __init__(
        self,
        expression_data: np.ndarray,
        cnv_data: np.ndarray,
        cell_ids: np.ndarray,
        metadata: Optional[pd.DataFrame] = None,
        n_hard_negatives: int = 5,
        cnv_similarity_threshold: float = 0.9,
        expression_dissimilarity_threshold: float = 0.5,
        precompute_negatives: bool = True
    ):
        """
        Args:
            expression_data: (n_cells, n_genes) expression matrix
            cnv_data: (n_cells, n_cnv_windows) CNV matrix
            n_hard_negatives: Number of hard negatives per sample
            cnv_similarity_threshold: Minimum CNV similarity to be a candidate
            expression_dissimilarity_threshold: Minimum expression difference to be hard negative
            precompute_negatives: Whether to precompute hard negatives (faster training)
        """
        assert len(expression_data) == len(cnv_data)

        self.expression = torch.FloatTensor(expression_data)
        self.cnv = torch.FloatTensor(cnv_data)
        self.cell_ids = cell_ids
        self.metadata = metadata
        self.n_hard_negatives = n_hard_negatives

        if precompute_negatives:
            print("Precomputing hard negatives...")
            self.hard_negative_indices = self._find_hard_negatives(
                expression_data, cnv_data,
                cnv_similarity_threshold,
                expression_dissimilarity_threshold
            )
            print(f"  Found hard negatives for {len(self.hard_negative_indices)} cells")
        else:
            self.hard_negative_indices = None

    def _find_hard_negatives(
        self,
        expression: np.ndarray,
        cnv: np.ndarray,
        cnv_sim_thresh: float,
        expr_dissim_thresh: float,
        max_candidates: int = 5000
    ) -> Dict[int, np.ndarray]:
        """
        Find hard negatives for each cell using memory-efficient sampling.

        Hard negative = cell with similar CNV but different expression

        For large datasets (>5000 cells), we sample a random subset of candidates
        to compare against rather than doing full pairwise comparisons.
        This reduces memory from O(n^2) to O(n * max_candidates).
        """
        n_cells = len(expression)
        hard_negatives = {}

        # Pre-normalize all data once
        print(f"    Normalizing {n_cells:,} cells...")
        all_cnv_norm = cnv / (np.linalg.norm(cnv, axis=1, keepdims=True) + 1e-8)
        all_expr_norm = expression / (np.linalg.norm(expression, axis=1, keepdims=True) + 1e-8)

        # Use sampling for large datasets to avoid OOM
        use_sampling = n_cells > max_candidates
        if use_sampling:
            print(f"    Using memory-efficient sampling mode ({max_candidates:,} candidates per batch)")
            print(f"    This is slower but won't run out of memory")

        # Smaller batches for memory efficiency
        batch_size = 200 if use_sampling else 1000
        n_batches = (n_cells + batch_size - 1) // batch_size

        cells_with_negatives = 0
        import time
        start_time = time.time()

        for batch_idx, i in enumerate(range(0, n_cells, batch_size)):
            batch_end = min(i + batch_size, n_cells)
            batch_indices = np.arange(i, batch_end)

            # Progress with ETA
            elapsed = time.time() - start_time
            if batch_idx > 0:
                eta = elapsed / batch_idx * (n_batches - batch_idx)
                eta_str = f"ETA: {eta/60:.1f}min"
            else:
                eta_str = "ETA: calculating..."

            print(f"    Batch {batch_idx + 1}/{n_batches} | "
                  f"Cells with negatives: {cells_with_negatives:,} | {eta_str}     ", end='\r')

            if use_sampling:
                # Sample random candidates (excluding current batch)
                all_indices = np.arange(n_cells)
                mask = ~np.isin(all_indices, batch_indices)
                available = all_indices[mask]

                if len(available) > max_candidates:
                    sample_idx = np.random.choice(available, max_candidates, replace=False)
                else:
                    sample_idx = available

                # Compute similarities only against sampled candidates
                cnv_sim = np.dot(all_cnv_norm[batch_indices], all_cnv_norm[sample_idx].T)
                expr_sim = np.dot(all_expr_norm[batch_indices], all_expr_norm[sample_idx].T)
                expr_dissim = 1 - expr_sim

                for j, idx in enumerate(batch_indices):
                    candidates_mask = (cnv_sim[j] > cnv_sim_thresh) & (expr_dissim[j] > expr_dissim_thresh)
                    candidates_local = np.where(candidates_mask)[0]

                    if len(candidates_local) > 0:
                        candidates_global = sample_idx[candidates_local]
                        ranked = candidates_global[np.argsort(-expr_dissim[j, candidates_local])]
                        hard_negatives[idx] = ranked[:self.n_hard_negatives * 2]
                        cells_with_negatives += 1
            else:
                # Full pairwise for smaller datasets
                cnv_sim = np.dot(all_cnv_norm[batch_indices], all_cnv_norm.T)
                expr_sim = np.dot(all_expr_norm[batch_indices], all_expr_norm.T)
                expr_dissim = 1 - expr_sim

                for j, idx in enumerate(batch_indices):
                    candidates = np.where(
                        (cnv_sim[j] > cnv_sim_thresh) &
                        (expr_dissim[j] > expr_dissim_thresh) &
                        (np.arange(n_cells) != idx)
                    )[0]

                    if len(candidates) > 0:
                        ranked = candidates[np.argsort(-expr_dissim[j, candidates])]
                        hard_negatives[idx] = ranked[:self.n_hard_negatives * 2]
                        cells_with_negatives += 1

        elapsed = time.time() - start_time
        print(f"\n    Done in {elapsed/60:.1f} minutes. Found hard negatives for {cells_with_negatives:,}/{n_cells:,} cells")
        return hard_negatives

    def __len__(self):
        return len(self.expression)

    def __getitem__(self, idx):
        item = {
            'expression': self.expression[idx],
            'cnv': self.cnv[idx],
            'idx': idx
        }

        # Add hard negatives if available
        if self.hard_negative_indices is not None and idx in self.hard_negative_indices:
            neg_indices = self.hard_negative_indices[idx]
            # Randomly sample n_hard_negatives
            if len(neg_indices) > self.n_hard_negatives:
                sampled = np.random.choice(neg_indices, self.n_hard_negatives, replace=False)
            else:
                sampled = neg_indices

            item['hard_neg_expression'] = self.expression[sampled]
            item['hard_neg_cnv'] = self.cnv[sampled]
            item['has_hard_negatives'] = True
        else:
            # No hard negatives found - use placeholder
            item['hard_neg_expression'] = torch.zeros(1, self.expression.shape[1])
            item['hard_neg_cnv'] = torch.zeros(1, self.cnv.shape[1])
            item['has_hard_negatives'] = False

        return item


# ============================================================================
# Model Architecture (same encoders, new loss)
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


class ContrastiveModelWithHardNegatives(nn.Module):
    """
    Contrastive model with hard negative mining.

    Loss = InfoNCE(positive pairs) + lambda * HardNegativeLoss

    The hard negative loss explicitly pushes apart:
    - Cells with similar CNV but different expression
    - This teaches the model to recognize when expression doesn't follow CNV
    """

    def __init__(
        self,
        expression_dim: int,
        cnv_dim: int,
        latent_dim: int = 128,
        expression_hidden: List[int] = [512, 256],
        cnv_hidden: List[int] = [256, 128],
        temperature: float = 0.07,
        dropout: float = 0.1,
        hard_negative_weight: float = 0.5,
        hard_negative_margin: float = 0.3
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
        self.hard_negative_weight = hard_negative_weight
        self.hard_negative_margin = hard_negative_margin

    def forward(self, expression, cnv):
        """Forward pass returning embeddings."""
        expression_embed = self.expression_encoder(expression)
        cnv_embed = self.cnv_encoder(cnv)
        return expression_embed, cnv_embed

    def compute_infonce_loss(self, expression_embed, cnv_embed):
        """Standard InfoNCE contrastive loss."""
        batch_size = expression_embed.shape[0]

        logits = torch.matmul(expression_embed, cnv_embed.T) / self.temperature
        labels = torch.arange(batch_size, device=logits.device)

        loss_exp_to_cnv = F.cross_entropy(logits, labels)
        loss_cnv_to_exp = F.cross_entropy(logits.T, labels)

        loss = (loss_exp_to_cnv + loss_cnv_to_exp) / 2

        with torch.no_grad():
            pred_exp = logits.argmax(dim=1)
            pred_cnv = logits.T.argmax(dim=1)
            acc_exp = (pred_exp == labels).float().mean()
            acc_cnv = (pred_cnv == labels).float().mean()
            accuracy = (acc_exp + acc_cnv) / 2

        return loss, accuracy

    def compute_hard_negative_loss(
        self,
        anchor_expr_embed: torch.Tensor,
        anchor_cnv_embed: torch.Tensor,
        hard_neg_expr_embed: torch.Tensor,
        hard_neg_cnv_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        Hard negative loss using triplet margin.

        For cells with similar CNV but different expression:
        - The anchor's expression embedding should be CLOSER to its own CNV embedding
        - Than to the hard negative's CNV embedding (which has similar CNV pattern)

        This teaches: even though CNVs are similar, the expression patterns differ,
        so the model should recognize this discordance.
        """
        # Positive distance: anchor expression to anchor CNV
        pos_dist = 1 - (anchor_expr_embed * anchor_cnv_embed).sum(dim=1)

        # Negative distance: anchor expression to hard negative CNV
        # (hard negative has similar CNV, so in a naive model these would be close)
        neg_dist = 1 - (anchor_expr_embed * hard_neg_cnv_embed).sum(dim=1)

        # Triplet loss: positive should be closer than negative by margin
        loss = F.relu(pos_dist - neg_dist + self.hard_negative_margin)

        return loss.mean()

    def compute_loss(
        self,
        expression_embed: torch.Tensor,
        cnv_embed: torch.Tensor,
        hard_neg_expr: Optional[torch.Tensor] = None,
        hard_neg_cnv: Optional[torch.Tensor] = None,
        has_hard_negatives: Optional[torch.Tensor] = None
    ):
        """
        Combined loss: InfoNCE + Hard Negative Loss
        """
        # Standard InfoNCE loss
        infonce_loss, accuracy = self.compute_infonce_loss(expression_embed, cnv_embed)

        # Hard negative loss (if provided)
        hard_neg_loss = torch.tensor(0.0, device=expression_embed.device)

        if hard_neg_expr is not None and has_hard_negatives is not None:
            # Only compute for samples that have hard negatives
            mask = has_hard_negatives.bool()

            if mask.sum() > 1:  # Need at least 2 for BatchNorm
                # Get anchors with hard negatives
                anchor_expr = expression_embed[mask]
                anchor_cnv = cnv_embed[mask]

                # Get hard negative embeddings
                # hard_neg_expr/cnv are (batch, n_negatives, dim)
                hn_expr = hard_neg_expr[mask]
                hn_cnv = hard_neg_cnv[mask]

                # Encode hard negatives
                batch_size, n_neg, expr_dim = hn_expr.shape
                hn_expr_flat = hn_expr.view(-1, expr_dim)
                hn_cnv_flat = hn_cnv.view(-1, hn_cnv.shape[-1])

                # Set to eval mode for encoding hard negatives (avoids BatchNorm issues with small batches)
                self.expression_encoder.eval()
                self.cnv_encoder.eval()

                with torch.no_grad():
                    hn_expr_embed = self.expression_encoder(hn_expr_flat)
                    hn_cnv_embed = self.cnv_encoder(hn_cnv_flat)

                # Back to train mode
                self.expression_encoder.train()
                self.cnv_encoder.train()

                hn_expr_embed = hn_expr_embed.view(batch_size, n_neg, -1)
                hn_cnv_embed = hn_cnv_embed.view(batch_size, n_neg, -1)

                # Compute hard negative loss for each negative
                total_hn_loss = 0
                for neg_idx in range(n_neg):
                    hn_loss = self.compute_hard_negative_loss(
                        anchor_expr,
                        anchor_cnv,
                        hn_expr_embed[:, neg_idx],
                        hn_cnv_embed[:, neg_idx]
                    )
                    total_hn_loss += hn_loss

                hard_neg_loss = total_hn_loss / n_neg

        # Combined loss
        total_loss = infonce_loss + self.hard_negative_weight * hard_neg_loss

        return total_loss, accuracy, infonce_loss.item(), hard_neg_loss.item()


# ============================================================================
# Data Loading (reuse from original)
# ============================================================================

def load_patient_data(
    patient_id: str,
    cnv_dir: str = 'data/cnv_output'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Load expression and CNV data for a patient."""
    cnv_path = os.path.join(cnv_dir, patient_id, f'{patient_id}_cnv.h5ad')
    if not os.path.exists(cnv_path):
        raise FileNotFoundError(f"CNV file not found: {cnv_path}")

    print(f"Loading {patient_id}...")
    adata = sc.read_h5ad(cnv_path)

    if sparse.issparse(adata.X):
        expression = adata.X.toarray()
    else:
        expression = np.array(adata.X)

    cnv = adata.obsm['X_cnv']
    if sparse.issparse(cnv):
        cnv = cnv.toarray()
    else:
        cnv = np.array(cnv)

    cell_ids = adata.obs_names.values
    metadata = adata.obs.copy()

    print(f"  Cells: {len(cell_ids):,}")
    print(f"  Expression dim: {expression.shape[1]:,}")
    print(f"  CNV dim: {cnv.shape[1]:,}")

    return expression, cnv, cell_ids, metadata


def load_all_patients(
    patient_ids: List[str],
    cnv_dir: str = 'data/cnv_output',
    max_cells_per_patient: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Load and concatenate data from multiple patients.

    Args:
        max_cells_per_patient: If set, randomly subsample to this many cells
                               per patient to reduce memory usage.
    """
    all_expression = []
    all_cnv = []
    all_cell_ids = []
    all_metadata = []

    for patient_id in patient_ids:
        try:
            expr, cnv, cells, meta = load_patient_data(patient_id, cnv_dir)

            # Subsample if requested
            if max_cells_per_patient is not None and len(cells) > max_cells_per_patient:
                idx = np.random.choice(len(cells), max_cells_per_patient, replace=False)
                expr = expr[idx]
                cnv = cnv[idx]
                cells = cells[idx]
                meta = meta.iloc[idx]
                print(f"    Subsampled to {max_cells_per_patient:,} cells")

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
    random_seed: int = 42,
    n_hard_negatives: int = 5,
    cnv_similarity_threshold: float = 0.9,
    expression_dissimilarity_threshold: float = 0.5,
    max_cells_per_patient: Optional[int] = None
) -> Dict:
    """Prepare train/val/test datasets with hard negative mining.

    Args:
        max_cells_per_patient: If set, subsample to this many cells per patient
                               to reduce memory usage. None = use all cells.
    """
    np.random.seed(random_seed)

    expression, cnv, cell_ids, metadata = load_all_patients(
        patient_ids, max_cells_per_patient=max_cells_per_patient
    )

    # Normalize
    expression_mean = expression.mean(axis=0)
    expression_std = expression.std(axis=0) + 1e-8
    expression = (expression - expression_mean) / expression_std

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
    print("\nCreating training dataset with hard negatives...")
    datasets = {
        'train': HardNegativeDataset(
            expression[train_idx],
            cnv[train_idx],
            cell_ids[train_idx],
            metadata.iloc[train_idx],
            n_hard_negatives=n_hard_negatives,
            cnv_similarity_threshold=cnv_similarity_threshold,
            expression_dissimilarity_threshold=expression_dissimilarity_threshold,
            precompute_negatives=True
        ),
        'val': HardNegativeDataset(
            expression[val_idx],
            cnv[val_idx],
            cell_ids[val_idx],
            metadata.iloc[val_idx],
            n_hard_negatives=n_hard_negatives,
            precompute_negatives=False  # Don't need for validation
        ),
        'test': HardNegativeDataset(
            expression[test_idx],
            cnv[test_idx],
            cell_ids[test_idx],
            metadata.iloc[test_idx],
            n_hard_negatives=n_hard_negatives,
            precompute_negatives=False
        )
    }

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

def collate_fn(batch):
    """Custom collate function to handle variable-size hard negatives."""
    expression = torch.stack([b['expression'] for b in batch])
    cnv = torch.stack([b['cnv'] for b in batch])
    idx = torch.tensor([b['idx'] for b in batch])
    has_hard_negatives = torch.tensor([b['has_hard_negatives'] for b in batch])

    # Pad hard negatives to same size
    max_neg = max(b['hard_neg_expression'].shape[0] for b in batch)

    hard_neg_expr_list = []
    hard_neg_cnv_list = []

    for b in batch:
        hn_expr = b['hard_neg_expression']
        hn_cnv = b['hard_neg_cnv']

        # Pad if needed
        if hn_expr.shape[0] < max_neg:
            pad_size = max_neg - hn_expr.shape[0]
            hn_expr = torch.cat([hn_expr, torch.zeros(pad_size, hn_expr.shape[1])])
            hn_cnv = torch.cat([hn_cnv, torch.zeros(pad_size, hn_cnv.shape[1])])

        hard_neg_expr_list.append(hn_expr)
        hard_neg_cnv_list.append(hn_cnv)

    hard_neg_expression = torch.stack(hard_neg_expr_list)
    hard_neg_cnv = torch.stack(hard_neg_cnv_list)

    return {
        'expression': expression,
        'cnv': cnv,
        'idx': idx,
        'hard_neg_expression': hard_neg_expression,
        'hard_neg_cnv': hard_neg_cnv,
        'has_hard_negatives': has_hard_negatives
    }


def train_epoch(
    model: ContrastiveModelWithHardNegatives,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float, float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_acc = 0
    total_infonce = 0
    total_hn = 0
    n_batches = 0

    for batch in dataloader:
        expression = batch['expression'].to(device)
        cnv = batch['cnv'].to(device)
        hard_neg_expr = batch['hard_neg_expression'].to(device)
        hard_neg_cnv = batch['hard_neg_cnv'].to(device)
        has_hard_negatives = batch['has_hard_negatives'].to(device)

        optimizer.zero_grad()

        exp_embed, cnv_embed = model(expression, cnv)
        loss, acc, infonce, hn_loss = model.compute_loss(
            exp_embed, cnv_embed,
            hard_neg_expr, hard_neg_cnv,
            has_hard_negatives
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += acc.item()
        total_infonce += infonce
        total_hn += hn_loss
        n_batches += 1

    return (total_loss / n_batches, total_acc / n_batches,
            total_infonce / n_batches, total_hn / n_batches)


def evaluate(
    model: ContrastiveModelWithHardNegatives,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[float, float]:
    """Evaluate model (InfoNCE only for validation)."""
    model.eval()
    total_loss = 0
    total_acc = 0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            expression = batch['expression'].to(device)
            cnv = batch['cnv'].to(device)

            exp_embed, cnv_embed = model(expression, cnv)
            loss, acc = model.compute_infonce_loss(exp_embed, cnv_embed)

            total_loss += loss.item()
            total_acc += acc.item()
            n_batches += 1

    return total_loss / n_batches, total_acc / n_batches


def train_model(
    datasets: Dict,
    output_dir: str = 'models/contrastive_hn',
    latent_dim: int = 128,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    n_epochs: int = 100,
    patience: int = 10,
    hard_negative_weight: float = 0.5,
    hard_negative_margin: float = 0.3,
    device: str = 'auto'
) -> ContrastiveModelWithHardNegatives:
    """Train the contrastive model with hard negative mining."""
    os.makedirs(output_dir, exist_ok=True)

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

    expression_dim = datasets['train'].expression.shape[1]
    cnv_dim = datasets['train'].cnv.shape[1]

    print(f"Expression dimension: {expression_dim}")
    print(f"CNV dimension: {cnv_dim}")
    print(f"Latent dimension: {latent_dim}")
    print(f"Hard negative weight: {hard_negative_weight}")
    print(f"Hard negative margin: {hard_negative_margin}")

    model = ContrastiveModelWithHardNegatives(
        expression_dim=expression_dim,
        cnv_dim=cnv_dim,
        latent_dim=latent_dim,
        hard_negative_weight=hard_negative_weight,
        hard_negative_margin=hard_negative_margin
    ).to(device)

    train_loader = DataLoader(
        datasets['train'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        datasets['val'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'train_infonce': [], 'train_hn': [],
        'val_loss': [], 'val_acc': []
    }

    print(f"\nTraining for up to {n_epochs} epochs...")
    print("-" * 80)

    for epoch in range(n_epochs):
        train_loss, train_acc, train_infonce, train_hn = train_epoch(
            model, train_loader, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, device)

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_infonce'].append(train_infonce)
        history['train_hn'].append(train_hn)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1:3d} | "
              f"Train Loss: {train_loss:.4f} (InfoNCE: {train_infonce:.4f}, HN: {train_hn:.4f}) "
              f"Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'expression_dim': expression_dim,
                'cnv_dim': cnv_dim,
                'latent_dim': latent_dim,
                'hard_negative_weight': hard_negative_weight,
                'hard_negative_margin': hard_negative_margin,
                'normalization': datasets['normalization']
            }, os.path.join(output_dir, 'best_model.pt'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)

    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"\nBest model from epoch {checkpoint['epoch']+1}")
    print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    print(f"  Val Accuracy: {checkpoint['val_acc']:.4f}")

    return model


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
        description='Train contrastive model WITH hard negative mining'
    )
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--patients', type=str, nargs='+', default=None)
    parser.add_argument('--output-dir', type=str, default='models/contrastive_hn')
    parser.add_argument('--latent-dim', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hard-negative-weight', type=float, default=0.5,
                        help='Weight for hard negative loss (0-1)')
    parser.add_argument('--hard-negative-margin', type=float, default=0.3,
                        help='Margin for triplet loss')
    parser.add_argument('--n-hard-negatives', type=int, default=5,
                        help='Number of hard negatives per sample')
    parser.add_argument('--cnv-sim-threshold', type=float, default=0.7,
                        help='Minimum CNV similarity for hard negative candidates')
    parser.add_argument('--expr-dissim-threshold', type=float, default=0.3,
                        help='Minimum expression dissimilarity for hard negatives')
    parser.add_argument('--max-cells-per-patient', type=int, default=None,
                        help='Subsample to N cells per patient to reduce memory (default: use all)')

    args = parser.parse_args()

    if args.patients:
        patient_ids = args.patients
    else:
        patient_ids = get_available_patients()

    if not patient_ids:
        print("No patients with CNV data found!")
        return

    print("=" * 70)
    print("Contrastive Learning with HARD NEGATIVE MINING")
    print("=" * 70)
    print(f"\nPatients: {patient_ids}")

    if args.train:
        print("\nPreparing data with hard negative mining...")
        if args.max_cells_per_patient:
            print(f"Subsampling to {args.max_cells_per_patient:,} cells per patient for memory efficiency")

        datasets = prepare_data(
            patient_ids,
            n_hard_negatives=args.n_hard_negatives,
            cnv_similarity_threshold=args.cnv_sim_threshold,
            expression_dissimilarity_threshold=args.expr_dissim_threshold,
            max_cells_per_patient=args.max_cells_per_patient
        )

        model = train_model(
            datasets,
            output_dir=args.output_dir,
            latent_dim=args.latent_dim,
            batch_size=args.batch_size,
            n_epochs=args.epochs,
            learning_rate=args.lr,
            hard_negative_weight=args.hard_negative_weight,
            hard_negative_margin=args.hard_negative_margin
        )

        print(f"\nModel saved to: {args.output_dir}")
        print("\nTo run DE analysis with this model:")
        print(f"  python 08_differential_expression.py --model-dir {args.output_dir} --all-patients")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
