"""
Run CNV inference using infercnvpy.

This script:
1. Loads preprocessed AnnData
2. Adds gene position annotations
3. Runs infercnvpy to infer CNV profiles
4. Clusters cells by CNV profiles to create subclusters
5. Saves CNV profiles per subcluster for downstream contrastive learning

Usage:
    python 05_run_infercnv.py --patient P0006
    python 05_run_infercnv.py --all-patients

Requirements:
    pip install infercnvpy
"""

import os
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Dict

try:
    import infercnvpy as cnv
    INFERCNVPY_AVAILABLE = True
except ImportError:
    INFERCNVPY_AVAILABLE = False
    print("Error: infercnvpy not installed. Install with: pip install infercnvpy")


# Patient sample mapping
PATIENT_SAMPLES = {
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


def load_gene_positions(
    gene_positions_file: str = "data/reference/gene_positions_hg38.tsv"
) -> pd.DataFrame:
    """Load gene positions from cached file."""
    if not os.path.exists(gene_positions_file):
        raise FileNotFoundError(
            f"Gene positions file not found: {gene_positions_file}\n"
            "Run: python 04_prepare_infercnv.py --patient P0006 first to generate it."
        )

    df = pd.read_csv(gene_positions_file, sep='\t')
    print(f"Loaded {len(df):,} gene positions")
    return df


def add_gene_positions_to_adata(
    adata: sc.AnnData,
    gene_positions: pd.DataFrame
) -> sc.AnnData:
    """
    Add chromosome and position info to adata.var for infercnvpy.

    infercnvpy requires:
    - adata.var['chromosome']: chromosome number (1-22, X, Y)
    - adata.var['start']: gene start position
    - adata.var['end']: gene end position
    """
    print("Adding gene position annotations to AnnData...")

    # Create mapping from gene name to position (drop duplicates first)
    gene_positions_unique = gene_positions.drop_duplicates(subset='gene', keep='first')
    gene_pos_dict = gene_positions_unique.set_index('gene').to_dict('index')

    # Add columns to adata.var
    chromosomes = []
    starts = []
    ends = []

    for gene in adata.var_names:
        if gene in gene_pos_dict:
            pos = gene_pos_dict[gene]
            # Add 'chr' prefix for infercnvpy compatibility
            chrom = pos['chr']
            if not str(chrom).startswith('chr'):
                chrom = f"chr{chrom}"
            chromosomes.append(chrom)
            starts.append(pos['start'])
            ends.append(pos['end'])
        else:
            chromosomes.append(None)
            starts.append(None)
            ends.append(None)

    adata.var['chromosome'] = chromosomes
    adata.var['start'] = starts
    adata.var['end'] = ends

    # Count genes with positions
    n_with_pos = adata.var['chromosome'].notna().sum()
    print(f"Genes with chromosome positions: {n_with_pos:,} / {adata.n_vars:,}")

    # Filter to genes with positions (required for infercnvpy)
    adata = adata[:, adata.var['chromosome'].notna()].copy()
    print(f"After filtering: {adata.n_vars:,} genes")

    # Convert chromosome to string (infercnvpy expects string, not categorical)
    adata.var['chromosome'] = adata.var['chromosome'].astype(str)

    # Convert start/end to int
    adata.var['start'] = adata.var['start'].astype(int)
    adata.var['end'] = adata.var['end'].astype(int)

    # Sort genes by chromosome and position (required by infercnvpy)
    chrom_order = {f'chr{i}': i for i in range(1, 23)}
    chrom_order['chrX'] = 23
    chrom_order['chrY'] = 24
    adata.var['chr_order'] = adata.var['chromosome'].map(chrom_order)
    gene_order = adata.var.sort_values(['chr_order', 'start']).index
    adata = adata[:, gene_order].copy()
    adata.var = adata.var.drop('chr_order', axis=1)

    print(f"Genes sorted by chromosomal position")

    return adata


def run_infercnv(
    adata: sc.AnnData,
    reference_key: str = 'cancer_vs_normal',
    reference_cat: str = 'Normal',
    window_size: int = 250,
    step: int = 10,
    dynamic_threshold: float = 1.5,
    leiden_resolution: float = 1.0,
    n_jobs: int = 1,
) -> sc.AnnData:
    """
    Run infercnvpy CNV inference pipeline.

    Args:
        adata: AnnData with gene positions in .var
        reference_key: Column in adata.obs containing reference/observation labels
        reference_cat: Category name for reference (normal) cells
        window_size: Number of genes per window for smoothing
        step: Step size for sliding window
        dynamic_threshold: Threshold for CNV calling
        leiden_resolution: Resolution for CNV-based clustering

    Returns:
        AnnData with CNV scores in .obsm['X_cnv'] and subclusters in .obs['cnv_leiden']
    """
    print("\n" + "=" * 60)
    print("Running infercnvpy CNV inference")
    print("=" * 60)

    # Step 1: Normalize and log-transform (required for infercnvpy)
    print("\n1. Preprocessing for CNV inference...")
    adata_cnv = adata.copy()

    # Store raw counts
    adata_cnv.layers['counts'] = adata_cnv.X.copy()

    # Normalize
    sc.pp.normalize_total(adata_cnv, target_sum=1e4)
    sc.pp.log1p(adata_cnv)

    # Step 2: Run infercnvpy
    print("\n2. Computing CNV scores...")
    print(f"   Reference group: '{reference_cat}' from '{reference_key}'")
    print(f"   Window size: {window_size} genes, step: {step}")

    # Check genes per chromosome
    chr_counts = adata_cnv.var['chromosome'].value_counts()
    print(f"   Genes per chromosome (min: {chr_counts.min()}, max: {chr_counts.max()})")

    # Filter chromosomes with too few genes for the window size
    valid_chroms = chr_counts[chr_counts >= window_size].index.tolist()
    if len(valid_chroms) < len(chr_counts):
        excluded = set(chr_counts.index) - set(valid_chroms)
        print(f"   Excluding chromosomes with < {window_size} genes: {excluded}")
        adata_cnv = adata_cnv[:, adata_cnv.var['chromosome'].isin(valid_chroms)].copy()
        print(f"   After filtering: {adata_cnv.n_vars} genes")

    cnv.tl.infercnv(
        adata_cnv,
        reference_key=reference_key,
        reference_cat=[reference_cat],
        window_size=window_size,
        step=step,
        exclude_chromosomes=('chrX', 'chrY'),
        n_jobs=n_jobs,
    )

    print(f"   CNV matrix shape: {adata_cnv.obsm['X_cnv'].shape}")

    # Step 3: Cluster cells by CNV profile (must happen before cnv_score)
    print("\n3. Clustering cells by CNV profile...")
    cnv.tl.pca(adata_cnv)
    cnv.pp.neighbors(adata_cnv)
    cnv.tl.leiden(adata_cnv, resolution=leiden_resolution)

    n_clusters = adata_cnv.obs['cnv_leiden'].nunique()
    print(f"   Found {n_clusters} CNV subclusters")

    # Step 4: Compute CNV scores per cell (requires cnv_leiden)
    print("\n4. Computing per-cell CNV scores...")
    cnv.tl.cnv_score(adata_cnv)
    print(f"   CNV scores range: [{adata_cnv.obs['cnv_score'].min():.3f}, {adata_cnv.obs['cnv_score'].max():.3f}]")

    # Step 5: Show cluster sizes
    print("\n   Subcluster sizes:")
    cluster_counts = adata_cnv.obs['cnv_leiden'].value_counts().sort_index()
    for cluster, count in cluster_counts.items():
        print(f"     Cluster {cluster}: {count:,} cells")

    return adata_cnv


def compute_cnv_profiles_per_subcluster(
    adata: sc.AnnData,
    subcluster_col: str = 'cnv_leiden'
) -> pd.DataFrame:
    """
    Compute mean CNV profile for each subcluster.

    These profiles will be used as anchors for contrastive learning.

    Args:
        adata: AnnData with CNV scores in .obsm['X_cnv']
        subcluster_col: Column containing subcluster labels

    Returns:
        DataFrame with shape (n_subclusters, n_cnv_windows)
    """
    print("\nComputing mean CNV profiles per subcluster...")

    subclusters = sorted(adata.obs[subcluster_col].unique())
    cnv_matrix = adata.obsm['X_cnv']

    profiles = []
    for cluster in subclusters:
        mask = (adata.obs[subcluster_col] == cluster).values  # Convert to numpy array
        mean_profile = cnv_matrix[mask].mean(axis=0)
        # Handle sparse matrix output
        if hasattr(mean_profile, 'A1'):
            mean_profile = mean_profile.A1
        profiles.append(np.asarray(mean_profile).flatten())

    # Create DataFrame
    cnv_profiles = pd.DataFrame(
        np.array(profiles),
        index=[f"subcluster_{c}" for c in subclusters]
    )

    print(f"CNV profiles shape: {cnv_profiles.shape}")

    return cnv_profiles


def plot_cnv_results(
    adata: sc.AnnData,
    output_dir: str,
    patient_id: str
) -> None:
    """Generate CNV visualization plots."""
    print("\nGenerating CNV plots...")

    os.makedirs(output_dir, exist_ok=True)

    # 1. CNV heatmap (let infercnvpy create its own figure)
    try:
        cnv.pl.chromosome_heatmap(adata, groupby='cnv_leiden', show=False)
        plt.title(f'{patient_id} - CNV Heatmap by Subcluster')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cnv_heatmap.png'), dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"   Warning: Could not generate CNV heatmap: {e}")

    # 2. CNV score distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # By cancer vs normal
    for label in ['Cancer', 'Normal']:
        mask = adata.obs['cancer_vs_normal'] == label
        axes[0].hist(adata.obs.loc[mask, 'cnv_score'], bins=50, alpha=0.6, label=label)
    axes[0].set_xlabel('CNV Score')
    axes[0].set_ylabel('Number of Cells')
    axes[0].set_title('CNV Score Distribution')
    axes[0].legend()

    # By subcluster
    subclusters = sorted(adata.obs['cnv_leiden'].unique())
    for cluster in subclusters[:10]:  # Show first 10 clusters
        mask = adata.obs['cnv_leiden'] == cluster
        axes[1].hist(adata.obs.loc[mask, 'cnv_score'], bins=30, alpha=0.5, label=f'Cluster {cluster}')
    axes[1].set_xlabel('CNV Score')
    axes[1].set_ylabel('Number of Cells')
    axes[1].set_title('CNV Score by Subcluster')
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cnv_score_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 3. PCA of CNV profiles
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        sc.pl.embedding(adata, basis='X_cnv_pca', color='cancer_vs_normal', ax=axes[0], show=False, title='CNV PCA - Cancer vs Normal')
        sc.pl.embedding(adata, basis='X_cnv_pca', color='cnv_leiden', ax=axes[1], show=False, title='CNV PCA - Subclusters')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cnv_pca.png'), dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"   Warning: Could not generate CNV PCA plot: {e}")

    print(f"Plots saved to {output_dir}")


def run_cnv_inference_for_patient(
    patient_id: str,
    input_dir: str = 'data/processed',
    output_dir: str = 'data/cnv_output',
    gene_positions_file: str = 'data/reference/gene_positions_hg38.tsv',
    leiden_resolution: float = 0.5,
    save_plots: bool = True
) -> Dict[str, any]:
    """
    Run full CNV inference pipeline for a patient.

    Args:
        patient_id: Patient ID (e.g., 'P0006')
        input_dir: Directory containing preprocessed .h5ad files
        output_dir: Directory to save CNV outputs
        gene_positions_file: Path to gene positions TSV
        leiden_resolution: Resolution for CNV clustering
        save_plots: Whether to generate plots

    Returns:
        Dictionary with paths to output files
    """
    if not INFERCNVPY_AVAILABLE:
        raise ImportError("infercnvpy not installed. Install with: pip install infercnvpy")

    print("\n" + "=" * 60)
    print(f"CNV INFERENCE FOR PATIENT {patient_id}")
    print("=" * 60)

    # Create output directory
    patient_output_dir = os.path.join(output_dir, patient_id)
    os.makedirs(patient_output_dir, exist_ok=True)

    # Load preprocessed data
    h5ad_path = os.path.join(input_dir, f'{patient_id}_preprocessed.h5ad')
    print(f"\nLoading {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    print(f"Loaded: {adata.shape[0]:,} cells x {adata.shape[1]:,} genes")

    # Load and add gene positions
    gene_positions = load_gene_positions(gene_positions_file)
    adata = add_gene_positions_to_adata(adata, gene_positions)

    # Run CNV inference
    adata = run_infercnv(
        adata,
        reference_key='cancer_vs_normal',
        reference_cat='Normal',
        leiden_resolution=leiden_resolution
    )

    # Compute CNV profiles per subcluster
    cnv_profiles = compute_cnv_profiles_per_subcluster(adata, subcluster_col='cnv_leiden')

    # Save outputs
    output_paths = {
        'adata': os.path.join(patient_output_dir, f'{patient_id}_cnv.h5ad'),
        'cnv_profiles': os.path.join(patient_output_dir, 'cnv_profiles.csv'),
        'cell_subclusters': os.path.join(patient_output_dir, 'cell_subclusters.csv'),
    }

    # Save AnnData with CNV results
    print(f"\nSaving outputs to {patient_output_dir}")
    adata.write_h5ad(output_paths['adata'])
    print(f"  Saved: {output_paths['adata']}")

    # Save CNV profiles
    cnv_profiles.to_csv(output_paths['cnv_profiles'])
    print(f"  Saved: {output_paths['cnv_profiles']}")

    # Save cell-to-subcluster mapping
    cell_subclusters = adata.obs[['cnv_leiden', 'cancer_vs_normal', 'cell_type']].copy()
    cell_subclusters.to_csv(output_paths['cell_subclusters'])
    print(f"  Saved: {output_paths['cell_subclusters']}")

    # Generate plots
    if save_plots:
        plot_dir = os.path.join(patient_output_dir, 'plots')
        plot_cnv_results(adata, plot_dir, patient_id)
        output_paths['plots'] = plot_dir

    # Summary
    print("\n" + "=" * 60)
    print("CNV INFERENCE COMPLETE")
    print("=" * 60)
    n_subclusters = adata.obs['cnv_leiden'].nunique()
    print(f"\nPatient: {patient_id}")
    print(f"Cells: {adata.n_obs:,}")
    print(f"CNV subclusters: {n_subclusters}")
    print(f"CNV profile shape: {cnv_profiles.shape}")

    return output_paths


def main():
    parser = argparse.ArgumentParser(
        description='Run CNV inference using infercnvpy'
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
        '--input-dir',
        type=str,
        default='data/processed',
        help='Directory containing preprocessed .h5ad files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/cnv_output',
        help='Directory to save CNV outputs'
    )
    parser.add_argument(
        '--leiden-resolution',
        type=float,
        default=2.0,
        help='Resolution for CNV-based clustering (default: 2.0)'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip plot generation'
    )

    args = parser.parse_args()

    # Determine which patients to process
    if args.all_patients:
        patients = list(PATIENT_SAMPLES.keys())
    else:
        patients = [args.patient]

    # Process each patient
    for patient_id in patients:
        if patient_id not in PATIENT_SAMPLES:
            print(f"Warning: Unknown patient {patient_id}, skipping")
            continue

        # Check if preprocessed file exists
        h5ad_path = os.path.join(args.input_dir, f'{patient_id}_preprocessed.h5ad')
        if not os.path.exists(h5ad_path):
            print(f"Warning: {h5ad_path} not found, skipping {patient_id}")
            continue

        try:
            run_cnv_inference_for_patient(
                patient_id=patient_id,
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                leiden_resolution=args.leiden_resolution,
                save_plots=not args.no_plots
            )
        except Exception as e:
            print(f"Error processing {patient_id}: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
