# Weakly Supervised Contrastive Alignment of scRNA-seq to CNV Anchors

This repository contains code to recreate the paper "Weakly Supervised Contrastive Alignment of scRNA-seq to CNV Anchors" from scratch.

## Overview

This project implements a contrastive learning framework that aligns single-cell RNA-seq expression profiles with inferred copy number variation (CNV) information. The key idea is to train a gene expression encoder to align with fixed CNV anchors in a shared latent space.

## Project Structure

```
project/
├── data/                      # Data directory
│   ├── raw/                   # Raw downloaded data
│   └── processed/             # Preprocessed data
├── notebooks/                 # Analysis notebooks
│   ├── 01_data_download.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_infercnv.ipynb
│   ├── 04_training.ipynb
│   └── 05_analysis.ipynb
├── src/                       # Source code
│   ├── model.py              # Encoder architectures
│   ├── losses.py             # Loss functions
│   ├── data_processing.py    # Data utilities
│   ├── train.py              # Training loop
│   └── evaluation.py         # Evaluation metrics
├── checkpoints/              # Model checkpoints
├── figures/                  # Generated figures
├── requirements.txt          # Dependencies
└── README.md                # This file
```

## Installation

```bash
# Create conda environment
conda create -n cnv_alignment python=3.10
conda activate cnv_alignment

# Install dependencies
pip install -r requirements.txt

# For inferCNV (R package), install separately:
# In R: install.packages("BiocManager")
#       BiocManager::install("infercnv")
```

## Quick Start

### 1. Data Download (Week 1)

Download Patient P0006 from GSE131907 (Kim et al. 2020 lung adenocarcinoma atlas):
- LUNG_T06 (tumor sample)
- LUNG_N06 (normal lung sample - used as reference)

```python
# See notebooks/01_data_download.ipynb
```

### 2. Preprocessing (Week 1)

```python
import scanpy as sc
from data_processing import preprocess_adata

# Load data
adata = sc.read_h5ad('data/raw/GSE131907_P0006.h5ad')

# Preprocess
adata = preprocess_adata(
    adata,
    min_genes=200,
    min_cells=3,
    target_sum=1e4,
    log_transform=True
)

# Expected: 5,884 genes after filtering
```

### 3. CNV Inference (Week 1-2)

Run inferCNV in R to generate CNV profiles and subclusters:

```R
library(infercnv)

# Create inferCNV object
infercnv_obj = CreateInfercnvObject(
    raw_counts_matrix="data/processed/counts_matrix.txt",
    annotations_file="data/processed/cell_annotations.txt",
    delim="\t",
    gene_order_file="data/processed/gene_positions.txt",
    ref_group_names=c("Normal")
)

# Run inferCNV
infercnv_obj = infercnv::run(
    infercnv_obj,
    cutoff=0.1,
    out_dir="data/processed/infercnv_output",
    cluster_by_groups=TRUE,
    denoise=TRUE,
    HMM=TRUE
)
```

Expected output: 78 CNV subclusters

### 4. Training (Week 2-3)

```python
from model import MultimodalEncoder
from data_processing import MultimodalScDataset
from train import train_model
import torch

# Load preprocessed data
adata = sc.read_h5ad('data/processed/adata_preprocessed.h5ad')
cnv_profiles = pd.read_csv('data/processed/cnv_profiles.csv', index_col=0)

# Create dataset
dataset = MultimodalScDataset(adata, cnv_profiles, subcluster_col='subcluster')

# Initialize model
model = MultimodalEncoder(n_genes=5884, hidden_dim=256, latent_dim=64)

# Train
model, history = train_model(
    model=model,
    dataset=dataset,
    cnv_profiles_tensor=torch.FloatTensor(cnv_profiles.values),
    n_epochs=100,
    batch_size=4096,
    learning_rate=1e-3,
    device='cuda'
)
```

### 5. Evaluation (Week 3-4)

```python
from evaluation import evaluate_alignment

# Evaluate alignment
results = evaluate_alignment(
    model=model,
    dataset=dataset,
    cnv_profiles_tensor=torch.FloatTensor(cnv_profiles.values),
    k=5
)

# Expected: 97.4% top-5 accuracy in z-space
print(f"Z-space accuracy: {results['z_space_accuracy']:.1%}")
```

## Implementation Details

### Model Architecture

- **Expression Encoder**: 3-layer MLP (5884 → 256 → 256 → 256)
- **CNV Encoder**: 3-layer MLP (5884 → 256 → 256 → 256) - **FROZEN**
- **Projection Heads**: 256 → 128 → 64 with L2 normalization
- Hidden dimension (h-space): 256
- Latent dimension (z-space): 64

### Loss Functions

1. **InfoNCE Contrastive Loss**: Aligns expression embeddings to CNV anchors
   - Temperature: τ = 0.2
   - Hard negative mining: top-k = 10

2. **Centroid Regularization**: Keeps cells close to subcluster centroids
   - Weight: λ_centroid = 0.05

3. **H-space Alignment**: Aligns intermediate representations
   - Weight: λ_h_align = 0.1

### Training Parameters

- Epochs: 100
- Batch size: 4,096 (large for contrastive learning)
- Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
- Gradient clipping: max_norm = 1.0

## Expected Results

### Alignment Performance
- **Z-space top-5 accuracy**: 97.4%
- **H-space top-5 accuracy**: ~20% (as reported)

### Visualizations
1. CNV similarity heatmap (Figure 3 in paper)
2. 3D UMAP of expression embeddings colored by cancer vs. normal
3. Training loss curves

### Biomarker Discovery
- Traditional DE: Epithelial cancer vs. normal cells
- CNV-conditioned DE: Within CNV neighborhoods

## Downstream Analysis

After training, the learned embeddings can be used for:

1. **Clustering**: Use h_expr for Leiden clustering
2. **Differential Expression**: CNV-aware DE analysis
3. **Visualization**: UMAP/t-SNE of embeddings
4. **Biomarker Discovery**: Identify cancer-specific markers

## Detailed Roadmap

### Phase 1: Data Setup (Week 1)
- [ ] Download GSE131907 Patient P0006 data
- [ ] Perform basic QC and filtering
- [ ] Create gene position file for inferCNV
- [ ] Verify matched tumor/normal samples

### Phase 2: CNV Inference (Week 1-2)
- [ ] Install and configure inferCNV
- [ ] Run inferCNV with normal as reference
- [ ] Extract 78 CNV subclusters
- [ ] Create subcluster annotations
- [ ] Generate CNV profile matrix

### Phase 3: Model Development (Week 2-3)
- [ ] Implement encoder architectures
- [ ] Implement all three loss functions
- [ ] Create PyTorch dataset class
- [ ] Test model with dummy data
- [ ] Verify gradient flow

### Phase 4: Training (Week 3)
- [ ] Set up training loop
- [ ] Implement hard negative mining
- [ ] Add logging and checkpointing
- [ ] Train for 100 epochs
- [ ] Monitor loss curves

### Phase 5: Evaluation (Week 3-4)
- [ ] Implement top-k retrieval metric
- [ ] Compute z-space and h-space accuracy
- [ ] Generate similarity heatmaps
- [ ] Create UMAP visualizations
- [ ] Compare to paper results (97.4%)

### Phase 6: Downstream Analysis (Week 4-5)
- [ ] Perform Leiden clustering on h_expr
- [ ] Traditional DE: cancer vs. normal epithelial
- [ ] CNV-conditioned DE analysis
- [ ] Identify top biomarkers
- [ ] Validate APOC1 downregulation finding

### Phase 7: Paper Writing (Week 5+)
- [ ] Document all results
- [ ] Create publication-quality figures
- [ ] Write methods section
- [ ] Compare findings to original paper

## Key Files

- `model.py`: Encoder and projection head architectures
- `losses.py`: InfoNCE, centroid, and h-space losses
- `data_processing.py`: Dataset class and preprocessing utilities
- `train.py`: Training loop with checkpointing
- `evaluation.py`: Top-k retrieval and visualization

## Troubleshooting

### Common Issues

1. **inferCNV fails**: Make sure normal reference is properly specified
2. **OOM during training**: Reduce batch size (try 2048 or 1024)
3. **Low accuracy**: Check that CNV encoder is frozen
4. **Slow training**: Enable GPU, use DataLoader with num_workers

## Citation

If you use this code, please cite the original paper:

```
Goswami, G., Xu, D., & Park, H. J. (2025). 
Weakly Supervised Contrastive Alignment of scRNA-seq to CNV Anchors.
```

## License

MIT License

## Contact

For questions or issues, please open a GitHub issue.
