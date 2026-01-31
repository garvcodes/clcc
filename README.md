# Contrastive Learning for CNV-Expression Concordance (CLCC)

A contrastive learning framework to identify genes that escape copy number variation (CNV) dosage effects in cancer. The model learns to align single-cell gene expression with CNV profiles, then identifies cells where expression diverges from expected CNV patterns - revealing potential escape mechanisms.

## Overview

**Key Idea**: In cancer, gene expression typically follows CNV dosage (amplified regions → higher expression). However, some genes "escape" this relationship through regulatory mechanisms. This project:

1. Trains a contrastive model to align expression with CNV in a shared latent space
2. Uses hard negative mining to improve discrimination
3. Classifies cells as "concordant" (expression follows CNV) or "discordant" (expression escapes CNV)
4. Performs differential expression to identify escape and compensation genes

## Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. Data Preparation                                                         │
│     00_download_data.py → 02_preprocess_raw_data.py                         │
│     Download GSE131907 lung adenocarcinoma data, preprocess per patient     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  2. CNV Inference                                                            │
│     04_prepare_infercnv.py → 05_run_infercnv.py                             │
│     Infer CNV profiles using normal cells as reference (inferCNV/R)         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  3. Contrastive Pre-training                                                 │
│     07_contrastive_model.py                                                  │
│     Train expression encoder to align with frozen CNV encoder               │
│     Loss: InfoNCE contrastive loss                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  4. Hard Negative Mining                                                     │
│     07b_contrastive_hard_negatives.py                                       │
│     Fine-tune with triplet loss on hard negatives                           │
│     (cells with similar CNV but different expression)                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  5. Concordance Classification                                               │
│     08_differential_expression.py                                            │
│     Compute expression-CNV embedding distance per cell                      │
│     Classify: Concordant (low distance) vs Discordant (high distance)       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  6. Differential Expression Analysis                                         │
│     08_differential_expression.py (per-patient)                              │
│     08c_pooled_de_analysis.py (pooled cross-patient)                        │
│     Identify escape genes (↑ in discordant) and compensation genes (↓)      │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Setup Environment

```bash
conda create -n clcc python=3.10
conda activate clcc
pip install -r requirements.txt
```

### 2. Download Data

```bash
python 00_download_data.py
```

Downloads GSE131907 lung adenocarcinoma atlas (~12GB).

### 3. Preprocess Data

```bash
# Single patient
python 02_preprocess_raw_data.py --patient P0006

# All patients
python 02_preprocess_raw_data.py --all-patients
```

### 4. Run CNV Inference

```bash
python 04_prepare_infercnv.py --patient P0006
python 05_run_infercnv.py --patient P0006
```

Requires R with inferCNV installed:
```R
BiocManager::install("infercnv")
```

### 5. Train Contrastive Model

```bash
# Base contrastive training
python 07_contrastive_model.py

# Hard negative fine-tuning (memory-efficient for local machines)
python 07b_contrastive_hard_negatives.py --max-cells-per-patient 5000
```

### 6. Run Differential Expression

```bash
# Per-patient DE with summary compilation
python 08_differential_expression.py --model-dir models/contrastive_hn --compile-summary

# Pooled cross-patient DE (more statistical power)
python 08c_pooled_de_analysis.py --model-dir models/contrastive_hn
```

## Project Structure

```
CLCC/
├── 00_download_data.py           # Download raw data from GEO
├── 02_preprocess_raw_data.py     # Preprocess scRNA-seq data
├── 04_prepare_infercnv.py        # Prepare inputs for inferCNV
├── 05_run_infercnv.py            # Run CNV inference
├── 07_contrastive_model.py       # Base contrastive training
├── 07b_contrastive_hard_negatives.py  # Hard negative mining
├── 08_differential_expression.py      # Per-patient DE analysis
├── 08b_standard_de.py            # Standard DE (cancer vs normal)
├── 08c_pooled_de_analysis.py     # Pooled cross-patient DE
├── notebooks/                    # Exploration notebooks
│   ├── 01_explore_raw_data.ipynb
│   ├── 03_explore_processed_data.ipynb
│   ├── 06_explore_example_cnv_results.ipynb
│   └── 09_explore_concordance_results.ipynb
├── data/                         # Data directory (gitignored)
│   ├── raw/                      # Raw GEO downloads
│   ├── processed/                # Preprocessed .h5ad files
│   ├── cnv_output/               # inferCNV results
│   └── de_results/               # DE analysis outputs
├── models/                       # Trained models (gitignored)
├── requirements.txt
└── README.md
```

## Key Outputs

After running the full pipeline, `data/de_results/` contains:

| File | Description |
|------|-------------|
| `pooled_de_results.csv` | Full DE results across all cells |
| `pooled_escape_genes.csv` | Genes higher in discordant cells (escape CNV) |
| `pooled_compensation_genes.csv` | Genes lower in discordant cells |
| `summary_all_patients.csv` | Per-patient summary statistics |
| `recurrent_genes.csv` | Genes significant in multiple patients |

## Model Architecture

- **Expression Encoder**: MLP (n_genes → 512 → 256 → 128)
- **CNV Encoder**: MLP (n_genes → 512 → 256 → 128) - **FROZEN**
- **Contrastive Loss**: InfoNCE + Triplet loss on hard negatives
- **Hard Negatives**: Cells with similar CNV but divergent expression

## Data

Uses GSE131907 (Kim et al. 2020) lung adenocarcinoma atlas with 10 matched tumor/normal patients:

| Patient | Tumor | Normal |
|---------|-------|--------|
| P0006 | LUNG_T06 | LUNG_N06 |
| P0008 | LUNG_T08 | LUNG_N08 |
| P0009 | LUNG_T09 | LUNG_N09 |
| P0018 | LUNG_T18 | LUNG_N18 |
| P0019 | LUNG_T19 | LUNG_N19 |
| P0020 | LUNG_T20 | LUNG_N20 |
| P0028 | LUNG_T28 | LUNG_N28 |
| P0030 | LUNG_T30 | LUNG_N30 |
| P0031 | LUNG_T31 | LUNG_N31 |
| P0034 | LUNG_T34 | LUNG_N34 |

## Requirements

Key dependencies (see `requirements.txt`):
- torch
- scanpy
- pandas, numpy
- scipy (for statistical tests)
- matplotlib, seaborn

## Citation

```
Goswami, G., Xu, D., & Park, H. J. (2025).
Contrastive Learning for CNV-Expression Concordance Analysis.
```

## License

MIT License
