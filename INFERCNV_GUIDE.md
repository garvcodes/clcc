# inferCNV Guide

This guide explains how to run inferCNV to generate CNV profiles and subclusters from scRNA-seq data.

## Installation

### In R:

```R
# Install BiocManager if needed
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

# Install inferCNV
BiocManager::install("infercnv")
```

## Required Input Files

inferCNV needs three input files:

### 1. Raw Counts Matrix (`counts_matrix.txt`)
Tab-delimited matrix with:
- Rows: genes
- Columns: cells
- Values: raw UMI counts (NOT normalized or log-transformed)

```python
# Generate from AnnData in Python
import pandas as pd
import scipy.sparse as sp

# Get raw counts (not normalized!)
if sp.issparse(adata.X):
    counts = pd.DataFrame(
        adata.X.toarray().T,  # Transpose: genes as rows
        index=adata.var_names,
        columns=adata.obs_names
    )
else:
    counts = pd.DataFrame(
        adata.X.T,
        index=adata.var_names,
        columns=adata.obs_names
    )

counts.to_csv('data/processed/counts_matrix.txt', sep='\t')
```

### 2. Cell Annotations (`cell_annotations.txt`)
Tab-delimited file with:
- Column 1: cell barcode
- Column 2: sample/group (e.g., "Normal" or "Tumor")

```python
# Generate from AnnData
annotations = adata.obs[['sample']].copy()  # or use cancer_vs_normal
annotations.to_csv('data/processed/cell_annotations.txt', sep='\t', header=False)
```

### 3. Gene Positions (`gene_positions.txt`)
Tab-delimited file with:
- Column 1: gene symbol
- Column 2: chromosome (no "chr" prefix)
- Column 3: start position
- Column 4: end position

```python
# Use the utility function
from data_processing import create_gene_position_file

create_gene_position_file(
    gene_list=adata.var_names.tolist(),
    output_path='data/processed/gene_positions.txt',
    assembly='hg38'
)
```

## Running inferCNV

### Basic Usage

```R
library(infercnv)

# Create inferCNV object
infercnv_obj = CreateInfercnvObject(
    raw_counts_matrix = "data/processed/counts_matrix.txt",
    annotations_file = "data/processed/cell_annotations.txt",
    delim = "\t",
    gene_order_file = "data/processed/gene_positions.txt",
    ref_group_names = c("Normal")  # Use normal cells as reference
)

# Run inferCNV
infercnv_obj = infercnv::run(
    infercnv_obj,
    cutoff = 0.1,  # Minimum expression cutoff
    out_dir = "data/processed/infercnv_output",
    cluster_by_groups = TRUE,  # Cluster cells
    denoise = TRUE,
    HMM = TRUE,  # Use HMM for CNV state calling
    num_threads = 8  # Adjust based on your system
)
```

### Important Parameters

- `cutoff`: Minimum expression threshold (default: 0.1)
- `cluster_by_groups`: Cluster cells into subclusters (TRUE for this project)
- `denoise`: Apply denoising (recommended: TRUE)
- `HMM`: Use Hidden Markov Model for CNV states (recommended: TRUE)
- `analysis_mode`: "subclusters" or "samples" (use "subclusters")

## Output Files

inferCNV creates several output files in the `out_dir`:

### Key Files:

1. **`infercnv.observations.txt`**: CNV matrix for tumor cells
   - Rows: genes
   - Columns: cells
   - Values: CNV scores (centered around 1.0)

2. **`map_metadata_from_infercnv.txt`**: Subcluster assignments
   - Maps each cell to its CNV subcluster

3. **`infercnv.png`**: Heatmap visualization of CNV profiles

## Extract Results for Python

### 1. Load Subcluster Assignments

```python
import pandas as pd

# Read subcluster map
subcluster_map = pd.read_csv(
    'data/processed/infercnv_output/map_metadata_from_infercnv.txt',
    sep='\t',
    header=None,
    names=['cell', 'subcluster']
)

# Add to AnnData
adata.obs['subcluster'] = adata.obs.index.map(
    subcluster_map.set_index('cell')['subcluster']
)

print(f"Found {adata.obs['subcluster'].nunique()} subclusters")
```

### 2. Create CNV Profile Matrix

```python
# Read CNV observations
cnv_obs = pd.read_csv(
    'data/processed/infercnv_output/infercnv.observations.txt',
    sep='\t',
    index_col=0
)

# Aggregate by subcluster (mean CNV score per subcluster)
cnv_profiles = []
subcluster_names = []

for subcluster in sorted(adata.obs['subcluster'].unique()):
    # Get cells in this subcluster
    cells_in_cluster = adata.obs[adata.obs['subcluster'] == subcluster].index
    
    # Get CNV scores for these cells
    cluster_cnv = cnv_obs[cells_in_cluster].mean(axis=1)
    
    cnv_profiles.append(cluster_cnv.values)
    subcluster_names.append(subcluster)

# Create DataFrame
cnv_profiles_df = pd.DataFrame(
    cnv_profiles,
    index=subcluster_names,
    columns=cnv_obs.index  # Gene names
)

# Align with adata gene order
cnv_profiles_df = cnv_profiles_df[adata.var_names]

# Save
cnv_profiles_df.to_csv('data/processed/cnv_profiles.csv')

print(f"CNV profiles shape: {cnv_profiles_df.shape}")
```

## Troubleshooting

### Common Issues:

1. **No reference cells found**
   - Check that `ref_group_names` matches your annotation file
   - Make sure normal cells are labeled correctly

2. **Out of memory**
   - Reduce number of cells (subset to tumor + some normals)
   - Reduce number of genes (use highly variable genes only)
   - Increase swap space

3. **Too few/many subclusters**
   - Adjust clustering parameters in `run()` function
   - Try different `cluster_by_groups` settings

4. **Gene position file errors**
   - Make sure chromosomes are numeric (1-22, X, Y)
   - Remove "chr" prefix
   - Check for duplicate genes

## Validation

After running inferCNV, validate the results:

```python
# Check subcluster distribution
print(adata.obs['subcluster'].value_counts())

# Visualize on UMAP
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.pl.umap(adata, color='subcluster')

# Check CNV profiles
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 8))
sns.heatmap(cnv_profiles_df, cmap='RdBu_r', center=1.0)
plt.title('CNV Profiles by Subcluster')
plt.tight_layout()
plt.show()
```

## Expected Results for Patient P0006

From the paper:
- **Total subclusters**: 78
- **Genes after filtering**: 5,884
- **Reference sample**: LUNG_N06 (normal lung)
- **Tumor sample**: LUNG_T06

## Next Steps

Once you have:
1. ✓ Subcluster assignments in `adata.obs['subcluster']`
2. ✓ CNV profile matrix (78 x 5884)

You can proceed to create the `MultimodalScDataset` and train the model!

## Additional Resources

- [inferCNV Documentation](https://github.com/broadinstitute/inferCNV/wiki)
- [inferCNV Tutorial](https://github.com/broadinstitute/inferCNV/wiki/Running-InferCNV)
- [Troubleshooting Guide](https://github.com/broadinstitute/inferCNV/wiki/Troubleshooting)
