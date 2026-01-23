"""
Preprocessing pipeline for raw scRNA-seq data prior to inferCNV.

This script handles:
1. Loading raw UMI count matrix and cell annotations from GSE131907
2. Filtering to specific patient(s) with matched tumor/normal samples
3. Data integrity checks
4. Merging annotation metadata into AnnData
5. Creating cancer_vs_normal binary labels
6. QC metrics calculation (mitochondrial genes, UMIs, detected genes)
7. QC visualization
8. Saving processed AnnData for downstream inferCNV analysis

Based on methodology from Kim et al. (2020) lung adenocarcinoma atlas.
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import warnings

# Configuration
#These 10 patients include both tumor and normal samples making them suitable for CNV analysis
PATIENT_SAMPLES = {
    # Patient ID: (tumor_sample, normal_sample)
    'P0006': ('LUNG_T06', 'LUNG_N06'),
    'P0008': ('LUNG_T08', 'LUNG_N08'),
    'P0009': ('LUNG_T09', 'LUNG_N09'),
    'P0018': ('LUNG_T18', 'LUNG_N18'),
    'P0019': ('LUNG_T19', 'LUNG_N19'),
    'P0020': ('LUNG_T20', 'LUNG_N20'),
    'P0028': ('LUNG_T28', 'LUNG_N28'),
    'P0030': ('LUNG_T30', 'LUNG_N30'),
    'P0031': ('LUNG_T31', 'LUNG_N31'),
    'P0034': ('LUNG_T34', 'LUNG_N34'),
}

# Sample origins considered "normal" for CNV inference reference
NORMAL_ORIGINS = {'nLung'}


def load_raw_umi_matrix(
    matrix_path: str,
    annotation_path: str,
    sample_ids: Optional[List[str]] = None,
    chunk_size: int = 10000
) -> Tuple[sc.AnnData, pd.DataFrame]:
    """
    Load raw UMI count matrix and filter to specific samples.

    The full matrix is large (~29K genes x 208K cells), so we load
    annotations first to identify relevant cell barcodes, then load
    only those columns from the matrix.

    Args:
        matrix_path: Path to GSE131907_Lung_Cancer_raw_UMI_matrix.txt
        annotation_path: Path to GSE131907_Lung_Cancer_cell_annotation.txt
        sample_ids: List of sample IDs to keep (e.g., ['LUNG_T06', 'LUNG_N06'])
                   If None, loads all samples (memory intensive!)
        chunk_size: Number of columns to read at a time

    Returns:
        Tuple of (AnnData object with raw counts, filtered annotations DataFrame)
    """
    print("=" * 60)
    print("Loading raw UMI matrix")
    print("=" * 60)

    # Load annotations first (small file)
    print(f"\nLoading cell annotations from: {annotation_path}")
    annotations = pd.read_csv(annotation_path, sep='\t')
    print(f"Total cells in annotation file: {len(annotations):,}")

    # Filter to requested samples
    if sample_ids is not None:
        mask = annotations['Sample'].isin(sample_ids)
        annotations = annotations[mask].copy()
        print(f"Cells after filtering to samples {sample_ids}: {len(annotations):,}")

    # Get list of cell indices to keep
    cell_indices = annotations['Index'].tolist()

    # Load the matrix - first just get the header to find column positions
    print(f"\nLoading UMI matrix from: {matrix_path}")
    print("Reading header to identify cell columns...")

    # Read just the header row
    with open(matrix_path, 'r') as f:
        header = f.readline().strip().split('\t')

    # First column is gene names, rest are cell indices
    all_cell_ids = header[1:]
    print(f"Total cells in matrix: {len(all_cell_ids):,}")

    # Find positions of cells we want to keep
    cell_id_set = set(cell_indices)
    cols_to_keep = [0]  # Always keep gene name column (index 0)
    kept_cell_ids = []

    for i, cell_id in enumerate(all_cell_ids):
        if cell_id in cell_id_set:
            cols_to_keep.append(i + 1)  # +1 because gene column is 0
            kept_cell_ids.append(cell_id)

    print(f"Cells to load: {len(kept_cell_ids):,}")

    # Load only the columns we need
    print("Loading filtered matrix (this may take a few minutes)...")
    df = pd.read_csv(
        matrix_path,
        sep='\t',
        usecols=cols_to_keep,
        index_col=0
    )

    print(f"Loaded matrix shape: {df.shape}")

    # Transpose so cells are rows, genes are columns
    print("Transposing matrix (cells x genes)...")
    df = df.T

    # Create AnnData object
    print("Creating AnnData object...")
    adata = sc.AnnData(
        X=df.values.astype(np.float32),
        obs=pd.DataFrame(index=df.index),
        var=pd.DataFrame(index=df.columns)
    )

    print(f"AnnData shape: {adata.shape[0]:,} cells x {adata.shape[1]:,} genes")

    return adata, annotations


def run_integrity_checks(adata: sc.AnnData) -> Dict[str, any]:
    """
    Perform basic integrity checks on the raw count matrix.

    Checks:
    1. Unique cell identifiers
    2. Unique gene identifiers
    3. Non-zero entry counts
    4. Non-integer values
    5. Negative values

    Args:
        adata: AnnData object with raw counts

    Returns:
        Dictionary with check results
    """
    print("\n" + "=" * 60)
    print("Running data integrity checks")
    print("=" * 60)

    results = {}

    # Check 1: Unique cell identifiers
    n_cells = adata.n_obs
    n_unique_cells = len(adata.obs_names.unique())
    results['unique_cells'] = n_cells == n_unique_cells
    print(f"\n1. Unique cell identifiers: {n_unique_cells:,} / {n_cells:,} - "
          f"{'PASS' if results['unique_cells'] else 'FAIL'}")

    # Check 2: Unique gene identifiers
    n_genes = adata.n_vars
    n_unique_genes = len(adata.var_names.unique())
    results['unique_genes'] = n_genes == n_unique_genes
    print(f"2. Unique gene identifiers: {n_unique_genes:,} / {n_genes:,} - "
          f"{'PASS' if results['unique_genes'] else 'FAIL'}")

    # Check 3: Non-zero entries
    if hasattr(adata.X, 'toarray'):
        X = adata.X.toarray()
    else:
        X = adata.X

    n_nonzero = np.count_nonzero(X)
    n_total = X.size
    sparsity = 1 - (n_nonzero / n_total)
    results['n_nonzero'] = n_nonzero
    results['sparsity'] = sparsity
    print(f"3. Non-zero entries: {n_nonzero:,} / {n_total:,} ({100 * (1 - sparsity):.2f}%)")
    print(f"   Sparsity: {100 * sparsity:.2f}%")

    # Check 4: Non-integer values
    is_integer = np.allclose(X, X.astype(int), equal_nan=True)
    results['all_integers'] = is_integer
    if not is_integer:
        non_int_count = np.sum(~np.isclose(X, X.astype(int)))
        print(f"4. Non-integer values found: {non_int_count:,} - WARNING")
    else:
        print(f"4. All values are integers: PASS")

    # Check 5: Negative values
    n_negative = np.sum(X < 0)
    results['no_negatives'] = n_negative == 0
    print(f"5. Negative values: {n_negative:,} - "
          f"{'PASS' if n_negative == 0 else 'FAIL'}")

    # Spot check - show sample of values
    print(f"\n   Spot check - value range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"   Mean: {X.mean():.4f}, Median: {np.median(X):.4f}")

    return results


def assess_sparsity(adata: sc.AnnData) -> Tuple[pd.Series, pd.Series]:
    """
    Assess data sparsity by quantifying zero-count cells and genes.

    Args:
        adata: AnnData object

    Returns:
        Tuple of (cells_per_gene, genes_per_cell) Series
    """
    print("\n" + "=" * 60)
    print("Assessing data sparsity")
    print("=" * 60)

    if hasattr(adata.X, 'toarray'):
        X = adata.X.toarray()
    else:
        X = adata.X

    # Genes with zero counts across all cells
    genes_detected_per_cell = np.sum(X > 0, axis=1)
    cells_detected_per_gene = np.sum(X > 0, axis=0)

    zero_count_genes = np.sum(cells_detected_per_gene == 0)
    zero_count_cells = np.sum(genes_detected_per_cell == 0)

    print(f"\nGenes with zero counts in all cells: {zero_count_genes:,} / {adata.n_vars:,}")
    print(f"Cells with zero detected genes: {zero_count_cells:,} / {adata.n_obs:,}")

    print(f"\nGenes detected per cell:")
    print(f"  Min: {genes_detected_per_cell.min():,}")
    print(f"  Max: {genes_detected_per_cell.max():,}")
    print(f"  Mean: {genes_detected_per_cell.mean():.1f}")
    print(f"  Median: {np.median(genes_detected_per_cell):.1f}")

    print(f"\nCells expressing each gene:")
    print(f"  Min: {cells_detected_per_gene.min():,}")
    print(f"  Max: {cells_detected_per_gene.max():,}")
    print(f"  Mean: {cells_detected_per_gene.mean():.1f}")
    print(f"  Median: {np.median(cells_detected_per_gene):.1f}")

    return (
        pd.Series(cells_detected_per_gene, index=adata.var_names),
        pd.Series(genes_detected_per_cell, index=adata.obs_names)
    )


def merge_annotations(
    adata: sc.AnnData,
    annotations: pd.DataFrame
) -> sc.AnnData:
    """
    Merge cell annotation metadata into AnnData object.

    Relevant fields:
    - Sample: sample ID
    - Sample_Origin: normal vs tumor tissue type
    - Cell_type: broad cell type labels
    - Cell_type.refined: refined cell type
    - Cell_subtype: specific subtype

    Also creates:
    - patient_id: extracted from Sample
    - cancer_vs_normal: binary label based on Sample_Origin

    Args:
        adata: AnnData object
        annotations: DataFrame with cell annotations

    Returns:
        AnnData with merged annotations in .obs
    """
    print("\n" + "=" * 60)
    print("Merging annotation metadata")
    print("=" * 60)

    # Set Index column as index for merging
    annotations = annotations.set_index('Index')

    # Ensure alignment with adata
    annotations = annotations.loc[adata.obs_names]

    # Add relevant columns to adata.obs
    adata.obs['barcode'] = annotations['Barcode']
    adata.obs['sample_id'] = annotations['Sample']
    adata.obs['sample_origin'] = annotations['Sample_Origin']
    adata.obs['cell_type'] = annotations['Cell_type']
    adata.obs['cell_type_refined'] = annotations['Cell_type.refined']
    adata.obs['cell_subtype'] = annotations['Cell_subtype']

    # Extract patient ID from sample name (e.g., LUNG_T06 -> P0006)
    def extract_patient_id(sample):
        # Extract numeric part from sample name
        parts = sample.split('_')
        if len(parts) >= 2:
            num = ''.join(filter(str.isdigit, parts[1]))
            if num:
                return f"P{num.zfill(4)}"
        return sample

    adata.obs['patient_id'] = adata.obs['sample_id'].apply(extract_patient_id)

    # Create cancer_vs_normal binary label
    # Normal: nLung (normal lung), nLN (normal lymph node)
    # Cancer: everything else (tLung, tL/B, mLN, PE, mBrain)
    adata.obs['cancer_vs_normal'] = adata.obs['sample_origin'].apply(
        lambda x: 'Normal' if x in NORMAL_ORIGINS else 'Cancer'
    )

    print(f"\nAnnotations merged successfully!")
    print(f"\nSample composition:")
    print(adata.obs['sample_id'].value_counts())

    print(f"\nSample origin distribution:")
    print(adata.obs['sample_origin'].value_counts())

    print(f"\nCancer vs Normal distribution:")
    print(adata.obs['cancer_vs_normal'].value_counts())

    print(f"\nCell type distribution:")
    print(adata.obs['cell_type'].value_counts())

    return adata


def calculate_qc_metrics(adata: sc.AnnData) -> sc.AnnData:
    """
    Calculate per-cell QC metrics using Scanpy.

    Metrics computed:
    - total_counts: total UMIs per cell
    - n_genes_by_counts: number of detected genes per cell
    - pct_counts_mt: percentage of mitochondrial counts

    Args:
        adata: AnnData object with raw counts

    Returns:
        AnnData with QC metrics in .obs and .var
    """
    print("\n" + "=" * 60)
    print("Calculating QC metrics")
    print("=" * 60)

    # Flag mitochondrial genes
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    n_mt_genes = adata.var['mt'].sum()
    print(f"\nMitochondrial genes identified: {n_mt_genes}")

    if n_mt_genes > 0:
        mt_genes = adata.var_names[adata.var['mt']].tolist()
        print(f"MT genes: {mt_genes[:10]}{'...' if len(mt_genes) > 10 else ''}")

    # Calculate QC metrics
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=['mt'],
        percent_top=None,
        log1p=False,
        inplace=True
    )

    print(f"\nQC metrics summary:")
    print(f"\nTotal counts (UMIs) per cell:")
    print(f"  Min: {adata.obs['total_counts'].min():.0f}")
    print(f"  Max: {adata.obs['total_counts'].max():.0f}")
    print(f"  Mean: {adata.obs['total_counts'].mean():.1f}")
    print(f"  Median: {adata.obs['total_counts'].median():.1f}")

    print(f"\nNumber of detected genes per cell:")
    print(f"  Min: {adata.obs['n_genes_by_counts'].min():.0f}")
    print(f"  Max: {adata.obs['n_genes_by_counts'].max():.0f}")
    print(f"  Mean: {adata.obs['n_genes_by_counts'].mean():.1f}")
    print(f"  Median: {adata.obs['n_genes_by_counts'].median():.1f}")

    print(f"\nMitochondrial percentage:")
    print(f"  Min: {adata.obs['pct_counts_mt'].min():.2f}%")
    print(f"  Max: {adata.obs['pct_counts_mt'].max():.2f}%")
    print(f"  Mean: {adata.obs['pct_counts_mt'].mean():.2f}%")
    print(f"  Median: {adata.obs['pct_counts_mt'].median():.2f}%")

    return adata


def plot_qc_metrics(
    adata: sc.AnnData,
    output_dir: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Generate QC visualization plots.

    Creates:
    1. Violin plots of QC metrics by sample
    2. Scatter plots: n_genes vs total_counts colored by pct_mt
    3. Distribution histograms

    Args:
        adata: AnnData object with QC metrics calculated
        output_dir: Directory to save plots (None = don't save)
        show: Whether to display plots
    """
    print("\n" + "=" * 60)
    print("Generating QC visualizations")
    print("=" * 60)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Set up style
    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. Violin plots by sample
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ['total_counts', 'n_genes_by_counts', 'pct_counts_mt']
    titles = ['Total UMIs', 'Detected Genes', 'Mitochondrial %']

    for ax, metric, title in zip(axes, metrics, titles):
        sc.pl.violin(
            adata,
            keys=metric,
            groupby='sample_id',
            ax=ax,
            show=False,
            rotation=45
        )
        ax.set_title(title)

    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'qc_violin_by_sample.png'), dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

    # 2. Violin plots by cancer vs normal
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    for ax, metric, title in zip(axes, metrics, titles):
        sc.pl.violin(
            adata,
            keys=metric,
            groupby='cancer_vs_normal',
            ax=ax,
            show=False
        )
        ax.set_title(title)

    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'qc_violin_cancer_vs_normal.png'), dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

    # 3. Scatter plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # n_genes vs total_counts
    scatter = axes[0].scatter(
        adata.obs['total_counts'],
        adata.obs['n_genes_by_counts'],
        c=adata.obs['pct_counts_mt'],
        cmap='viridis',
        alpha=0.5,
        s=1
    )
    axes[0].set_xlabel('Total UMIs')
    axes[0].set_ylabel('Detected Genes')
    axes[0].set_title('Genes vs UMIs (colored by MT%)')
    plt.colorbar(scatter, ax=axes[0], label='MT%')

    # Colored by cancer vs normal
    colors = {'Normal': 'blue', 'Cancer': 'red'}
    for label, color in colors.items():
        mask = adata.obs['cancer_vs_normal'] == label
        axes[1].scatter(
            adata.obs.loc[mask, 'total_counts'],
            adata.obs.loc[mask, 'n_genes_by_counts'],
            c=color,
            alpha=0.3,
            s=1,
            label=label
        )
    axes[1].set_xlabel('Total UMIs')
    axes[1].set_ylabel('Detected Genes')
    axes[1].set_title('Genes vs UMIs by Condition')
    axes[1].legend()

    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'qc_scatter.png'), dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

    # 4. Distribution histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(adata.obs['total_counts'], bins=100, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Total UMIs')
    axes[0].set_ylabel('Number of Cells')
    axes[0].set_title('Distribution of Total UMIs')
    axes[0].axvline(adata.obs['total_counts'].median(), color='red', linestyle='--', label='Median')
    axes[0].legend()

    axes[1].hist(adata.obs['n_genes_by_counts'], bins=100, color='darkorange', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Detected Genes')
    axes[1].set_ylabel('Number of Cells')
    axes[1].set_title('Distribution of Detected Genes')
    axes[1].axvline(adata.obs['n_genes_by_counts'].median(), color='red', linestyle='--', label='Median')
    axes[1].legend()

    axes[2].hist(adata.obs['pct_counts_mt'], bins=100, color='forestgreen', edgecolor='black', alpha=0.7)
    axes[2].set_xlabel('Mitochondrial %')
    axes[2].set_ylabel('Number of Cells')
    axes[2].set_title('Distribution of MT%')
    axes[2].axvline(adata.obs['pct_counts_mt'].median(), color='red', linestyle='--', label='Median')
    axes[2].legend()

    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'qc_distributions.png'), dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

    print("QC visualizations complete!")


def preprocess_patient(
    patient_id: str,
    data_dir: str = 'data/raw',
    output_dir: str = 'data/processed',
    plot_qc: bool = True,
    save: bool = True
) -> sc.AnnData:
    """
    Full preprocessing pipeline for a single patient.

    Args:
        patient_id: Patient ID (e.g., 'P0006')
        data_dir: Directory containing raw data files
        output_dir: Directory to save processed data
        plot_qc: Whether to generate QC plots
        save: Whether to save the processed AnnData

    Returns:
        Preprocessed AnnData object
    """
    print("\n" + "=" * 60)
    print(f"PREPROCESSING PATIENT {patient_id}")
    print("=" * 60)

    if patient_id not in PATIENT_SAMPLES:
        raise ValueError(f"Unknown patient ID: {patient_id}. "
                        f"Available patients: {list(PATIENT_SAMPLES.keys())}")

    tumor_sample, normal_sample = PATIENT_SAMPLES[patient_id]
    sample_ids = [tumor_sample, normal_sample]

    print(f"\nPatient {patient_id}:")
    print(f"  Tumor sample: {tumor_sample}")
    print(f"  Normal sample: {normal_sample}")

    # File paths
    matrix_path = os.path.join(data_dir, 'GSE131907_Lung_Cancer_raw_UMI_matrix.txt')
    annotation_path = os.path.join(data_dir, 'GSE131907_Lung_Cancer_cell_annotation.txt')

    # Load data
    adata, annotations = load_raw_umi_matrix(
        matrix_path=matrix_path,
        annotation_path=annotation_path,
        sample_ids=sample_ids
    )

    # Run integrity checks
    integrity_results = run_integrity_checks(adata)

    # Assess sparsity
    cells_per_gene, genes_per_cell = assess_sparsity(adata)

    # Merge annotations
    adata = merge_annotations(adata, annotations)

    # Calculate QC metrics
    adata = calculate_qc_metrics(adata)

    # Store processing info
    adata.uns['preprocessing'] = {
        'patient_id': patient_id,
        'tumor_sample': tumor_sample,
        'normal_sample': normal_sample,
        'integrity_checks': integrity_results,
        'n_cells_tumor': int((adata.obs['sample_id'] == tumor_sample).sum()),
        'n_cells_normal': int((adata.obs['sample_id'] == normal_sample).sum()),
    }

    # Generate QC plots
    if plot_qc:
        qc_plot_dir = os.path.join(output_dir, f'{patient_id}_qc_plots')
        plot_qc_metrics(adata, output_dir=qc_plot_dir, show=False)
        print(f"\nQC plots saved to: {qc_plot_dir}")

    # Save processed data
    if save:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{patient_id}_preprocessed.h5ad')
        adata.write_h5ad(output_path)
        print(f"\nProcessed data saved to: {output_path}")

    # Final summary
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"\nFinal AnnData shape: {adata.shape[0]:,} cells x {adata.shape[1]:,} genes")
    print(f"\nobs columns: {list(adata.obs.columns)}")
    print(f"var columns: {list(adata.var.columns)}")

    return adata


def preprocess_all_patients(
    data_dir: str = 'data/raw',
    output_dir: str = 'data/processed',
    patients: Optional[List[str]] = None,
    plot_qc: bool = True
) -> Dict[str, sc.AnnData]:
    """
    Preprocess multiple patients.

    Args:
        data_dir: Directory containing raw data files
        output_dir: Directory to save processed data
        patients: List of patient IDs to process (None = all available)
        plot_qc: Whether to generate QC plots

    Returns:
        Dictionary mapping patient ID to preprocessed AnnData
    """
    if patients is None:
        patients = list(PATIENT_SAMPLES.keys())

    results = {}
    for patient_id in patients:
        try:
            adata = preprocess_patient(
                patient_id=patient_id,
                data_dir=data_dir,
                output_dir=output_dir,
                plot_qc=plot_qc,
                save=True
            )
            results[patient_id] = adata
        except Exception as e:
            print(f"\nERROR processing patient {patient_id}: {e}")
            continue

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Preprocess raw scRNA-seq data for inferCNV analysis'
    )
    parser.add_argument(
        '--patient',
        type=str,
        default='P0006',
        help='Patient ID to process (default: P0006)'
    )
    parser.add_argument(
        '--all-patients',
        action='store_true',
        help='Process all available patients'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/raw',
        help='Directory containing raw data files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Directory to save processed data'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip QC plot generation'
    )

    args = parser.parse_args()

    if args.all_patients:
        preprocess_all_patients(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            plot_qc=not args.no_plots
        )
    else:
        preprocess_patient(
            patient_id=args.patient,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            plot_qc=not args.no_plots
        )
