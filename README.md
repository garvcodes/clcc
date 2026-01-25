# Weakly Supervised Contrastive Alignment of scRNA-seq to CNV Anchors

This repository contains code to recreate the paper "Weakly Supervised Contrastive Alignment of scRNA-seq to CNV Anchors" from scratch.

## Overview

This project implements a contrastive learning framework that aligns single-cell RNA-seq expression profiles with inferred copy number variation (CNV) information. The key idea is to train a gene expression encoder to align with fixed CNV anchors in a shared latent space.

## Quick Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/CLCC.git
cd CLCC
```

### 2. Create environment and install dependencies

```bash
conda create -n clcc python=3.10
conda activate clcc
pip install -r requirements.txt
```

### 3. Download the data

The raw data (~12GB) is hosted on NCBI GEO. Run the download script:

```bash
python download_data.py
```

This downloads from GSE131907 (Kim et al. 2020 lung adenocarcinoma atlas):
- `GSE131907_Lung_Cancer_raw_UMI_matrix.txt` (~12GB uncompressed)
- `GSE131907_Lung_Cancer_cell_annotation.txt` (~19MB)
- `GSE131907_Lung_Cancer_Feature_Summary.xlsx` (~20KB)

### 4. Preprocess the data

Process a single patient (P0006 as proof of concept):

```bash
python 01_preprocess_raw_data.py --patient P0006
```

Or process all 10 matched tumor/normal patients:

```bash
python 01_preprocess_raw_data.py --all-patients
```

This creates preprocessed `.h5ad` files in `data/processed/`.

### 5. Install inferCNV (R package)

```R
install.packages("BiocManager")
BiocManager::install("infercnv")
```

## Project Structure

```
CLCC/
├── data/
│   ├── raw/                    # Raw data (downloaded from GEO)
│   └── processed/              # Preprocessed .h5ad files
├── notebooks/
│   ├── 00_explore_raw_data.ipynb
│   └── 01_explore_processed_data.ipynb
├── 01_preprocess_raw_data.py   # Preprocessing pipeline
├── 01_data_processing.py       # Dataset classes and utilities
├── model.py                    # Encoder architectures
├── losses.py                   # Loss functions
├── train.py                    # Training loop
├── evaluation.py               # Evaluation metrics
├── download_data.py            # Data download script
├── requirements.txt
└── README.md
```

## Data

This project uses the GSE131907 dataset containing scRNA-seq from lung adenocarcinoma patients.

**Matched tumor/normal patients available:**
| Patient | Tumor Sample | Normal Sample |
|---------|--------------|---------------|
| P0006   | LUNG_T06     | LUNG_N06      |
| P0008   | LUNG_T08     | LUNG_N08      |
| P0009   | LUNG_T09     | LUNG_N09      |
| P0018   | LUNG_T18     | LUNG_N18      |
| P0019   | LUNG_T19     | LUNG_N19      |
| P0020   | LUNG_T20     | LUNG_N20      |
| P0028   | LUNG_T28     | LUNG_N28      |
| P0030   | LUNG_T30     | LUNG_N30      |
| P0031   | LUNG_T31     | LUNG_N31      |
| P0034   | LUNG_T34     | LUNG_N34      |

## Pipeline Overview

1. **Preprocessing** (`01_preprocess_raw_data.py`)
   - Load raw UMI counts for specific patient
   - Run data integrity checks
   - Merge cell annotations
   - Calculate QC metrics (UMIs, genes, MT%)
   - Create `cancer_vs_normal` labels
   - Save as `.h5ad` for downstream analysis

2. **CNV Inference** (inferCNV in R)
   - Use normal cells as reference
   - Generate CNV profiles per subcluster
   - Output: CNV matrix for contrastive learning

3. **Model Training** (`train.py`)
   - Expression encoder learns to align with CNV anchors
   - InfoNCE contrastive loss + centroid regularization
   - CNV encoder is frozen

4. **Evaluation** (`evaluation.py`)
   - Top-k retrieval accuracy in z-space
   - Expected: ~97% accuracy

## Model Architecture

- **Expression Encoder**: 3-layer MLP (n_genes → 256 → 256 → 256)
- **CNV Encoder**: 3-layer MLP (n_genes → 256 → 256 → 256) - **FROZEN**
- **Projection Heads**: 256 → 128 → 64 with L2 normalization

## Requirements

See `requirements.txt`. Key dependencies:
- scanpy
- torch
- pandas
- numpy
- matplotlib
- seaborn

## Citation

```
Goswami, G., Xu, D., & Park, H. J. (2025).
Weakly Supervised Contrastive Alignment of scRNA-seq to CNV Anchors.
```

## License

MIT License
