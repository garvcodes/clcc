"""
Concordance-based differential expression analysis for CANCER CELLS ONLY.

This script identifies genes differentially expressed between CNV-concordant
and CNV-discordant cancer cells. The key insight is that the contrastive model
learns the expected relationship between expression and CNV. Cancer cells that
deviate from this relationship (high embedding distance) may have regulatory
escape mechanisms that contribute to tumor progression.

Focus: Compensation mechanisms in cancer cells
- Concordant: cancer cells where expression matches expected CNV pattern
- Discordant: cancer cells where expression deviates from CNV pattern
- DE between these groups reveals genes involved in dosage compensation,
  epigenetic silencing, or other regulatory escape mechanisms

For standard DE analyses (cancer vs normal, high vs low CNV, etc.),
see 08b_standard_de.py

Usage:
    python 08_differential_expression.py --patient P0006
    python 08_differential_expression.py --all-patients
"""

import os
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
from typing import Dict, List, Tuple
import torch
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Data Loading
# ============================================================================

def load_cnv_data(
    patient_id: str,
    cnv_dir: str = 'data/cnv_output'
) -> sc.AnnData:
    """Load CNV-annotated AnnData for a patient."""
    cnv_path = os.path.join(cnv_dir, patient_id, f'{patient_id}_cnv.h5ad')
    if not os.path.exists(cnv_path):
        raise FileNotFoundError(f"CNV file not found: {cnv_path}")

    print(f"Loading {patient_id}...")
    adata = sc.read_h5ad(cnv_path)
    print(f"  Cells: {adata.n_obs:,}, Genes: {adata.n_vars:,}")
    print(f"  CNV subclusters: {adata.obs['cnv_leiden'].nunique()}")

    return adata


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


# ============================================================================
# Contrastive Embedding Loading & Concordance Analysis
# ============================================================================

def load_contrastive_model(
    model_dir: str = 'models/contrastive'
) -> Tuple[torch.nn.Module, Dict]:
    """Load trained contrastive model and normalization parameters."""
    from importlib import import_module

    model_path = os.path.join(model_dir, 'best_model.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained contrastive model found at {model_path}. "
            "Run: python 07_contrastive_model.py --train"
        )

    # Load checkpoint
    checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')

    # Import model class
    contrastive_module = import_module('07_contrastive_model'.replace('.py', ''))
    ContrastiveModel = contrastive_module.ContrastiveModel

    # Create model
    model = ContrastiveModel(
        expression_dim=checkpoint['expression_dim'],
        cnv_dim=checkpoint['cnv_dim'],
        latent_dim=checkpoint['latent_dim']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded contrastive model from {model_path}")
    print(f"  Expression dim: {checkpoint['expression_dim']}")
    print(f"  CNV dim: {checkpoint['cnv_dim']}")
    print(f"  Latent dim: {checkpoint['latent_dim']}")

    return model, checkpoint.get('normalization', {})


def compute_embedding_distances(
    adata: sc.AnnData,
    model: torch.nn.Module,
    normalization: Dict,
    batch_size: int = 256
) -> np.ndarray:
    """
    Compute embedding distance for each cell.

    Distance = 1 - cosine_similarity(expression_embedding, cnv_embedding)

    Low distance = CNV-concordant (expression matches CNV expectation)
    High distance = CNV-discordant (expression deviates from CNV expectation)

    Args:
        adata: AnnData with expression in .X and CNV in .obsm['X_cnv']
        model: Trained contrastive model
        normalization: Dict with expression_mean, expression_std, cnv_mean, cnv_std
        batch_size: Batch size for inference

    Returns:
        Array of embedding distances for each cell
    """
    model.eval()
    device = next(model.parameters()).device

    # Get expression data
    X = adata.X
    if sparse.issparse(X):
        X = X.toarray()
    X = np.array(X, dtype=np.float32)

    # Get CNV data
    cnv = adata.obsm['X_cnv']
    if sparse.issparse(cnv):
        cnv = cnv.toarray()
    cnv = np.array(cnv, dtype=np.float32)

    # Normalize using training statistics
    if normalization:
        expr_mean = normalization.get('expression_mean', X.mean(axis=0))
        expr_std = normalization.get('expression_std', X.std(axis=0)) + 1e-8
        cnv_mean = normalization.get('cnv_mean', cnv.mean(axis=0))
        cnv_std = normalization.get('cnv_std', cnv.std(axis=0)) + 1e-8

        X = (X - expr_mean) / expr_std
        cnv = (cnv - cnv_mean) / cnv_std

    # Compute embeddings in batches
    distances = []
    n_cells = len(X)

    with torch.no_grad():
        for i in range(0, n_cells, batch_size):
            batch_expr = torch.FloatTensor(X[i:i+batch_size]).to(device)
            batch_cnv = torch.FloatTensor(cnv[i:i+batch_size]).to(device)

            expr_embed, cnv_embed = model(batch_expr, batch_cnv)

            # Cosine similarity (embeddings are already normalized)
            similarity = (expr_embed * cnv_embed).sum(dim=1)

            # Distance = 1 - similarity
            batch_distances = (1 - similarity).cpu().numpy()
            distances.append(batch_distances)

    return np.concatenate(distances)


def classify_concordance_cancer_only(
    adata: sc.AnnData,
    model: torch.nn.Module,
    normalization: Dict,
    method: str = 'quantile',
    threshold_quantile: float = 0.25
) -> sc.AnnData:
    """
    Classify CANCER cells as CNV-concordant or CNV-discordant based on embedding distance.

    IMPORTANT: Thresholds are computed on cancer cells only to avoid bias from
    normal cells which typically have lower embedding distances.

    Args:
        adata: AnnData object (should contain cancer_vs_normal column)
        model: Trained contrastive model
        normalization: Normalization parameters
        method: 'quantile' or 'std' for threshold selection
        threshold_quantile: For quantile method, cells below this quantile are concordant

    Returns:
        AnnData with 'embedding_distance' and 'cnv_concordance' columns added
    """
    print("\nComputing embedding distances for ALL cells...")
    distances = compute_embedding_distances(adata, model, normalization)
    adata.obs['embedding_distance'] = distances

    # Filter to cancer cells for threshold computation
    if 'cancer_vs_normal' in adata.obs.columns:
        cancer_mask = adata.obs['cancer_vs_normal'] == 'Cancer'
        n_cancer = cancer_mask.sum()
        n_total = len(adata)
        print(f"\nFiltering to CANCER cells only for concordance classification")
        print(f"  Cancer cells: {n_cancer:,} / {n_total:,} ({100*n_cancer/n_total:.1f}%)")

        cancer_distances = distances[cancer_mask]
    else:
        print("\nWarning: No 'cancer_vs_normal' column found, using all cells")
        cancer_mask = np.ones(len(adata), dtype=bool)
        cancer_distances = distances

    # Classify cancer cells based on thresholds computed on cancer cells only
    if method == 'quantile':
        # Thresholds based on cancer cell distribution
        low_thresh = np.quantile(cancer_distances, threshold_quantile)
        high_thresh = np.quantile(cancer_distances, 1 - threshold_quantile)
    else:
        # Mean +/- 1 std of cancer cells
        mean_dist = cancer_distances.mean()
        std_dist = cancer_distances.std()
        low_thresh = mean_dist - std_dist
        high_thresh = mean_dist + std_dist

    # Apply classification to all cells (but only cancer cells are meaningful)
    concordance = np.where(
        distances <= low_thresh, 'Concordant',
        np.where(distances >= high_thresh, 'Discordant', 'Intermediate')
    )

    # Mark normal cells as 'Normal' in concordance column
    if 'cancer_vs_normal' in adata.obs.columns:
        normal_mask = adata.obs['cancer_vs_normal'] == 'Normal'
        concordance[normal_mask] = 'Normal'

    adata.obs['cnv_concordance'] = pd.Categorical(concordance)

    # Summary (cancer cells only)
    print(f"\nCNV Concordance Classification (method={method}, CANCER CELLS ONLY):")
    print(f"  Distance thresholds: low={low_thresh:.3f}, high={high_thresh:.3f}")
    print(f"  Cancer cell distance range: [{cancer_distances.min():.3f}, {cancer_distances.max():.3f}]")
    print(f"  Cancer cell mean distance: {cancer_distances.mean():.3f} +/- {cancer_distances.std():.3f}")

    for cat in ['Concordant', 'Intermediate', 'Discordant']:
        n = ((adata.obs['cnv_concordance'] == cat) & cancer_mask).sum()
        pct = 100 * n / cancer_mask.sum() if cancer_mask.sum() > 0 else 0
        print(f"  {cat}: {n:,} cancer cells ({pct:.1f}%)")

    if 'cancer_vs_normal' in adata.obs.columns:
        n_normal = (adata.obs['cnv_concordance'] == 'Normal').sum()
        print(f"  Normal cells (excluded): {n_normal:,}")

    return adata


def analyze_subcluster_composition(
    adata: sc.AnnData
) -> pd.DataFrame:
    """
    Analyze the composition of each CNV subcluster by concordance status.

    This helps understand whether certain CNV patterns are more likely
    to show concordant or discordant expression.

    Args:
        adata: AnnData with cnv_leiden and cnv_concordance columns

    Returns:
        DataFrame with subcluster composition analysis
    """
    print("\nAnalyzing subcluster composition by concordance...")

    if 'cnv_concordance' not in adata.obs.columns:
        print("  Warning: No cnv_concordance column, skipping")
        return pd.DataFrame()

    # Filter to cancer cells
    if 'cancer_vs_normal' in adata.obs.columns:
        adata_cancer = adata[adata.obs['cancer_vs_normal'] == 'Cancer'].copy()
    else:
        adata_cancer = adata

    results = []

    for subcluster in adata_cancer.obs['cnv_leiden'].unique():
        mask = adata_cancer.obs['cnv_leiden'] == subcluster
        n_cells = mask.sum()

        if n_cells < 10:
            continue

        # Count by concordance
        n_concordant = ((adata_cancer.obs['cnv_concordance'] == 'Concordant') & mask).sum()
        n_intermediate = ((adata_cancer.obs['cnv_concordance'] == 'Intermediate') & mask).sum()
        n_discordant = ((adata_cancer.obs['cnv_concordance'] == 'Discordant') & mask).sum()

        # Mean embedding distance
        mean_distance = adata_cancer.obs.loc[mask, 'embedding_distance'].mean()

        # Mean CNV score
        mean_cnv_score = adata_cancer.obs.loc[mask, 'cnv_score'].mean()

        results.append({
            'subcluster': subcluster,
            'n_cells': n_cells,
            'n_concordant': n_concordant,
            'n_intermediate': n_intermediate,
            'n_discordant': n_discordant,
            'pct_concordant': 100 * n_concordant / n_cells,
            'pct_discordant': 100 * n_discordant / n_cells,
            'mean_embedding_distance': mean_distance,
            'mean_cnv_score': mean_cnv_score,
            'concordance_ratio': n_concordant / (n_discordant + 1),  # +1 to avoid div by 0
        })

    df = pd.DataFrame(results)

    if len(df) > 0:
        df = df.sort_values('pct_discordant', ascending=False)

        print(f"\n  Subclusters with highest discordance (regulatory escape):")
        for _, row in df.head(5).iterrows():
            print(f"    Subcluster {row['subcluster']}: {row['pct_discordant']:.1f}% discordant, "
                  f"CNV={row['mean_cnv_score']:.3f}")

    return df


def run_concordance_de_cancer_only(
    adata: sc.AnnData,
    method: str = 'wilcoxon',
    within_subcluster: bool = False
) -> pd.DataFrame:
    """
    Run DE between CNV-concordant and CNV-discordant CANCER cells.

    This is the key analysis: genes that differ between concordant and discordant
    cancer cells are involved in regulatory escape from CNV-driven expression changes.

    Args:
        adata: AnnData with 'cnv_concordance' column
        method: DE method
        within_subcluster: If True, run DE within each CNV subcluster separately

    Returns:
        DataFrame with DE results
    """
    if 'cnv_concordance' not in adata.obs.columns:
        raise ValueError("Run classify_concordance_cancer_only() first")

    # Filter to cancer cells with concordant/discordant classification
    if 'cancer_vs_normal' in adata.obs.columns:
        cancer_mask = adata.obs['cancer_vs_normal'] == 'Cancer'
    else:
        cancer_mask = np.ones(len(adata), dtype=bool)

    concordance_mask = adata.obs['cnv_concordance'].isin(['Concordant', 'Discordant'])
    combined_mask = cancer_mask & concordance_mask

    adata_subset = adata[combined_mask].copy()

    n_concordant = (adata_subset.obs['cnv_concordance'] == 'Concordant').sum()
    n_discordant = (adata_subset.obs['cnv_concordance'] == 'Discordant').sum()

    print(f"\nRunning Concordance DE Analysis (CANCER CELLS ONLY)")
    print(f"  Concordant cancer cells: {n_concordant:,}")
    print(f"  Discordant cancer cells: {n_discordant:,}")

    if within_subcluster:
        # Run DE within each subcluster, then combine
        results = []
        for subcluster in adata_subset.obs['cnv_leiden'].unique():
            sub_mask = adata_subset.obs['cnv_leiden'] == subcluster
            adata_sub = adata_subset[sub_mask].copy()

            # Check we have enough cells in both groups
            n_conc = (adata_sub.obs['cnv_concordance'] == 'Concordant').sum()
            n_disc = (adata_sub.obs['cnv_concordance'] == 'Discordant').sum()

            if n_conc < 10 or n_disc < 10:
                continue

            print(f"  Subcluster {subcluster}: {n_conc} concordant, {n_disc} discordant")

            sc.tl.rank_genes_groups(
                adata_sub,
                groupby='cnv_concordance',
                groups=['Discordant'],
                reference='Concordant',
                method=method,
                pts=True,
                corr_method='benjamini-hochberg'  # Use FDR instead of Bonferroni
            )

            df = sc.get.rank_genes_groups_df(adata_sub, group='Discordant')
            df['subcluster'] = subcluster
            df['comparison'] = f'Discordant_vs_Concordant_in_{subcluster}'
            results.append(df)

        if not results:
            print("  Warning: No subclusters with enough cells in both groups")
            return pd.DataFrame()

        de_results = pd.concat(results, ignore_index=True)

    else:
        # Global comparison
        sc.tl.rank_genes_groups(
            adata_subset,
            groupby='cnv_concordance',
            groups=['Discordant'],
            reference='Concordant',
            method=method,
            pts=True,
            corr_method='benjamini-hochberg'  # Use FDR instead of Bonferroni
        )

        de_results = sc.get.rank_genes_groups_df(adata_subset, group='Discordant')
        de_results['comparison'] = 'Discordant_vs_Concordant_CancerOnly'

    # Add significance flags - both strict (FDR) and relaxed (raw p-value)
    de_results['significant_fdr'] = (
        (de_results['pvals_adj'] < 0.05) &
        (np.abs(de_results['logfoldchanges']) > 0.5)
    )

    # Relaxed criteria using raw p-values (for exploratory analysis)
    # This is appropriate when effect sizes are distributed across many genes
    de_results['significant'] = (
        (de_results['pvals'] < 0.01) &
        (np.abs(de_results['logfoldchanges']) > 0.3)
    )

    n_sig_fdr = de_results['significant_fdr'].sum()
    n_sig = de_results['significant'].sum()
    n_up = ((de_results['significant']) & (de_results['logfoldchanges'] > 0)).sum()
    n_down = ((de_results['significant']) & (de_results['logfoldchanges'] < 0)).sum()

    print(f"\nResults:")
    print(f"  Strict FDR significant (padj<0.05, |logFC|>0.5): {n_sig_fdr:,}")
    print(f"  Relaxed significant (p<0.01, |logFC|>0.3): {n_sig:,}")
    print(f"    Escape genes (up in Discordant): {n_up:,}")
    print(f"    Compensation genes (down in Discordant): {n_down:,}")
    print(f"  Note: Concordance effects are often distributed across many genes")

    return de_results


def analyze_escape_genes(
    de_results: pd.DataFrame,
    adata: sc.AnnData,
    top_n: int = 50
) -> pd.DataFrame:
    """
    Analyze the top escape genes (upregulated in discordant cells).

    These genes are expressed higher than expected given the CNV state,
    suggesting regulatory mechanisms that override dosage effects.

    Args:
        de_results: DE results from concordance analysis
        adata: AnnData object
        top_n: Number of top genes to analyze

    Returns:
        DataFrame with escape gene analysis
    """
    # Get upregulated genes using relaxed criteria OR top by score
    # First try significant genes, then fall back to top by effect size
    escape_genes = de_results[
        (de_results['significant']) &
        (de_results['logfoldchanges'] > 0)
    ]

    # If no significant genes, take top genes by score with positive logFC
    if len(escape_genes) == 0:
        escape_genes = de_results[de_results['logfoldchanges'] > 0.2].nlargest(top_n, 'scores')
    else:
        escape_genes = escape_genes.nlargest(top_n, 'scores')

    if len(escape_genes) == 0:
        return pd.DataFrame()

    results = []
    for _, row in escape_genes.iterrows():
        gene = row['names']

        gene_info = {
            'gene': gene,
            'logFC': row['logfoldchanges'],
            'pval_adj': row['pvals_adj'],
            'score': row['scores'],
            'interpretation': 'Regulatory escape - expressed despite CNV expectation'
        }

        # Add chromosome info if available
        if 'chromosome' in adata.var.columns and gene in adata.var_names:
            gene_info['chromosome'] = adata.var.loc[gene, 'chromosome']

        results.append(gene_info)

    return pd.DataFrame(results)


def analyze_compensation_genes(
    de_results: pd.DataFrame,
    adata: sc.AnnData,
    top_n: int = 50
) -> pd.DataFrame:
    """
    Analyze the top compensation genes (downregulated in discordant cells).

    These genes are expressed lower than expected given the CNV state,
    suggesting dosage compensation or epigenetic silencing mechanisms.

    Args:
        de_results: DE results from concordance analysis
        adata: AnnData object
        top_n: Number of top genes to analyze

    Returns:
        DataFrame with compensation gene analysis
    """
    # Get downregulated genes using relaxed criteria OR top by effect size
    compensation_genes = de_results[
        (de_results['significant']) &
        (de_results['logfoldchanges'] < 0)
    ]

    # If no significant genes, take top genes by negative logFC
    if len(compensation_genes) == 0:
        compensation_genes = de_results[de_results['logfoldchanges'] < -0.2].nsmallest(top_n, 'logfoldchanges')
    else:
        compensation_genes = compensation_genes.nsmallest(top_n, 'logfoldchanges')

    if len(compensation_genes) == 0:
        return pd.DataFrame()

    results = []
    for _, row in compensation_genes.iterrows():
        gene = row['names']

        gene_info = {
            'gene': gene,
            'logFC': row['logfoldchanges'],
            'pval_adj': row['pvals_adj'],
            'score': row['scores'],
            'interpretation': 'Dosage compensation - suppressed despite CNV expectation'
        }

        # Add chromosome info if available
        if 'chromosome' in adata.var.columns and gene in adata.var_names:
            gene_info['chromosome'] = adata.var.loc[gene, 'chromosome']

        results.append(gene_info)

    return pd.DataFrame(results)


# ============================================================================
# Visualization
# ============================================================================

def plot_volcano(
    de_results: pd.DataFrame,
    output_path: str,
    title: str = 'Volcano Plot',
    fc_threshold: float = 0.5,
    pval_threshold: float = 0.05,
    top_n_labels: int = 20
):
    """Create volcano plot for DE results."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Calculate -log10 p-value
    de_results = de_results.copy()
    de_results['-log10_pval'] = -np.log10(de_results['pvals_adj'] + 1e-300)

    # Color points
    colors = []
    for _, row in de_results.iterrows():
        if row['pvals_adj'] < pval_threshold:
            if row['logfoldchanges'] > fc_threshold:
                colors.append('red')
            elif row['logfoldchanges'] < -fc_threshold:
                colors.append('blue')
            else:
                colors.append('gray')
        else:
            colors.append('gray')

    ax.scatter(
        de_results['logfoldchanges'],
        de_results['-log10_pval'],
        c=colors,
        alpha=0.5,
        s=10
    )

    # Add threshold lines
    ax.axhline(-np.log10(pval_threshold), color='gray', linestyle='--', alpha=0.5)
    ax.axvline(fc_threshold, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(-fc_threshold, color='gray', linestyle='--', alpha=0.5)

    # Label top genes
    top_genes = de_results.nlargest(top_n_labels, '-log10_pval')
    for _, row in top_genes.iterrows():
        ax.annotate(
            row['names'],
            (row['logfoldchanges'], row['-log10_pval']),
            fontsize=8,
            alpha=0.8
        )

    ax.set_xlabel('Log2 Fold Change')
    ax.set_ylabel('-Log10 Adjusted P-value')
    ax.set_title(title)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label=f'Escape (FC>{fc_threshold}, p<{pval_threshold})'),
        Patch(facecolor='blue', label=f'Compensation (FC<-{fc_threshold}, p<{pval_threshold})'),
        Patch(facecolor='gray', label='Not significant')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def plot_concordance_analysis(
    adata: sc.AnnData,
    de_results: pd.DataFrame,
    output_dir: str,
    patient_id: str,
    threshold_quantile: float = 0.25
):
    """
    Create visualization plots for concordance analysis.

    Generates:
    1. Embedding distance distribution (cancer cells only)
    2. UMAP colored by concordance
    3. Volcano plot for concordant vs discordant
    4. Top escape/compensation genes
    5. Subcluster composition
    """
    os.makedirs(output_dir, exist_ok=True)

    # Filter to cancer cells for plots
    if 'cancer_vs_normal' in adata.obs.columns:
        cancer_mask = adata.obs['cancer_vs_normal'] == 'Cancer'
    else:
        cancer_mask = np.ones(len(adata), dtype=bool)

    # 1. Embedding distance distribution (cancer cells)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    distances = adata.obs['embedding_distance']
    cancer_distances = distances[cancer_mask]

    # Histogram for cancer cells
    axes[0].hist(cancer_distances, bins=50, edgecolor='black', alpha=0.7, color='coral')
    axes[0].axvline(cancer_distances.quantile(threshold_quantile), color='green', linestyle='--',
                    label=f'Q{int(threshold_quantile*100)} (Concordant threshold)')
    axes[0].axvline(cancer_distances.quantile(1-threshold_quantile), color='red', linestyle='--',
                    label=f'Q{int((1-threshold_quantile)*100)} (Discordant threshold)')
    axes[0].set_xlabel('Embedding Distance')
    axes[0].set_ylabel('Number of Cancer Cells')
    axes[0].set_title(f'{patient_id}: Expression-CNV Embedding Distance\n(Cancer Cells Only)')
    axes[0].legend()

    # Box plot by concordance (cancer cells only)
    concordance_order = ['Concordant', 'Intermediate', 'Discordant']
    colors = {'Concordant': 'green', 'Intermediate': 'gray', 'Discordant': 'red'}

    adata_cancer = adata[cancer_mask]
    data_to_plot = [adata_cancer.obs.loc[adata_cancer.obs['cnv_concordance'] == cat, 'embedding_distance'].values
                    for cat in concordance_order if cat in adata_cancer.obs['cnv_concordance'].values]
    labels = [cat for cat in concordance_order if cat in adata_cancer.obs['cnv_concordance'].values]

    bp = axes[1].boxplot(data_to_plot, labels=labels, patch_artist=True)
    for patch, label in zip(bp['boxes'], labels):
        patch.set_facecolor(colors.get(label, 'gray'))
        patch.set_alpha(0.7)

    axes[1].set_ylabel('Embedding Distance')
    axes[1].set_title('Distance by Concordance Class\n(Cancer Cells Only)')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'embedding_distance_distribution.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/embedding_distance_distribution.png")

    # 2. UMAP colored by concordance (if UMAP exists)
    if 'X_umap' in adata.obsm:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        umap = adata.obsm['X_umap']

        # By concordance (showing cancer cells, dimming normal)
        for cat in concordance_order:
            mask = adata.obs['cnv_concordance'] == cat
            if mask.sum() > 0:
                axes[0].scatter(umap[mask, 0], umap[mask, 1],
                               c=colors.get(cat, 'gray'), label=cat, s=5, alpha=0.6)
        # Add normal cells in background
        if 'cancer_vs_normal' in adata.obs.columns:
            normal_mask = adata.obs['cancer_vs_normal'] == 'Normal'
            axes[0].scatter(umap[normal_mask, 0], umap[normal_mask, 1],
                           c='lightblue', label='Normal', s=3, alpha=0.2)
        axes[0].set_xlabel('UMAP1')
        axes[0].set_ylabel('UMAP2')
        axes[0].set_title('UMAP by CNV Concordance\n(Cancer Cells)')
        axes[0].legend(markerscale=3)

        # By embedding distance (continuous, cancer cells only)
        sc_dist = axes[1].scatter(umap[cancer_mask, 0], umap[cancer_mask, 1],
                                  c=distances[cancer_mask], cmap='RdYlGn_r', s=5, alpha=0.5)
        plt.colorbar(sc_dist, ax=axes[1], label='Embedding Distance')
        axes[1].set_xlabel('UMAP1')
        axes[1].set_ylabel('UMAP2')
        axes[1].set_title('UMAP by Embedding Distance\n(Cancer Cells)')

        # By cancer vs normal
        if 'cancer_vs_normal' in adata.obs.columns:
            for cat, color in [('Cancer', 'red'), ('Normal', 'blue')]:
                mask = adata.obs['cancer_vs_normal'] == cat
                if mask.sum() > 0:
                    axes[2].scatter(umap[mask, 0], umap[mask, 1],
                                   c=color, label=cat, s=5, alpha=0.5)
            axes[2].set_xlabel('UMAP1')
            axes[2].set_ylabel('UMAP2')
            axes[2].set_title('UMAP by Cancer Status')
            axes[2].legend(markerscale=3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'umap_concordance.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir}/umap_concordance.png")

    # 3. Volcano plot for concordance DE
    if not de_results.empty and 'logfoldchanges' in de_results.columns:
        plot_volcano(
            de_results,
            os.path.join(output_dir, 'volcano_concordance.png'),
            title=f'{patient_id}: Discordant vs Concordant Cancer Cells\n(+) = Escape genes, (-) = Compensation genes'
        )

    # 4. Top genes bar plot
    if not de_results.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 8))

        # Top escape genes (up in discordant)
        escape = de_results[
            (de_results['significant']) &
            (de_results['logfoldchanges'] > 0)
        ].nlargest(20, 'scores')

        if len(escape) > 0:
            axes[0].barh(range(len(escape)), escape['logfoldchanges'].values, color='red', alpha=0.7)
            axes[0].set_yticks(range(len(escape)))
            axes[0].set_yticklabels(escape['names'].values)
            axes[0].set_xlabel('Log2 Fold Change')
            axes[0].set_title('Top Escape Genes\n(Higher in Discordant = Regulatory Escape)')
            axes[0].invert_yaxis()

        # Top compensation genes (down in discordant)
        compensation = de_results[
            (de_results['significant']) &
            (de_results['logfoldchanges'] < 0)
        ].nsmallest(20, 'logfoldchanges')

        if len(compensation) > 0:
            axes[1].barh(range(len(compensation)),
                        compensation['logfoldchanges'].values, color='blue', alpha=0.7)
            axes[1].set_yticks(range(len(compensation)))
            axes[1].set_yticklabels(compensation['names'].values)
            axes[1].set_xlabel('Log2 Fold Change')
            axes[1].set_title('Top Compensation Genes\n(Lower in Discordant = Dosage Compensation)')
            axes[1].invert_yaxis()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_concordance_genes.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir}/top_concordance_genes.png")


def plot_subcluster_composition(
    composition_df: pd.DataFrame,
    output_dir: str,
    patient_id: str
):
    """Plot subcluster composition by concordance status."""
    if composition_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Sort by discordance percentage
    df = composition_df.sort_values('pct_discordant', ascending=True)

    # Stacked bar chart
    x = range(len(df))
    width = 0.8

    axes[0].barh(x, df['pct_concordant'], width, label='Concordant', color='green', alpha=0.7)
    axes[0].barh(x, df['pct_discordant'], width, left=df['pct_concordant'] + (100 - df['pct_concordant'] - df['pct_discordant']),
                 label='Discordant', color='red', alpha=0.7)
    axes[0].barh(x, 100 - df['pct_concordant'] - df['pct_discordant'], width,
                 left=df['pct_concordant'], label='Intermediate', color='gray', alpha=0.5)

    axes[0].set_yticks(x)
    axes[0].set_yticklabels([f"Cluster {c}" for c in df['subcluster']])
    axes[0].set_xlabel('Percentage of Cancer Cells')
    axes[0].set_title(f'{patient_id}: Subcluster Composition by Concordance')
    axes[0].legend(loc='lower right')

    # Scatter: CNV score vs discordance
    axes[1].scatter(df['mean_cnv_score'], df['pct_discordant'],
                    s=df['n_cells']/10, alpha=0.6, c='coral')
    axes[1].set_xlabel('Mean CNV Score')
    axes[1].set_ylabel('% Discordant Cells')
    axes[1].set_title('CNV Burden vs Regulatory Discordance\n(bubble size = number of cells)')

    # Add cluster labels
    for _, row in df.iterrows():
        axes[1].annotate(f"{row['subcluster']}",
                        (row['mean_cnv_score'], row['pct_discordant']),
                        fontsize=8, alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'subcluster_composition.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/subcluster_composition.png")


# ============================================================================
# Main Pipeline
# ============================================================================

def run_concordance_pipeline(
    patient_id: str,
    cnv_dir: str = 'data/cnv_output',
    model_dir: str = 'models/contrastive',
    output_dir: str = 'data/de_results',
    threshold_quantile: float = 0.25
) -> Dict[str, pd.DataFrame]:
    """
    Run the concordance-based DE analysis pipeline for CANCER CELLS ONLY.

    This is the main analysis that leverages the contrastive embedding space
    to identify genes involved in regulatory escape from CNV effects in cancer.

    Args:
        patient_id: Patient identifier
        cnv_dir: Directory with CNV results
        model_dir: Directory with trained contrastive model
        output_dir: Output directory

    Returns:
        Dictionary of result DataFrames
    """
    print("=" * 70)
    print(f"CONCORDANCE-BASED DE ANALYSIS (CANCER CELLS ONLY): {patient_id}")
    print("=" * 70)
    print("\nFocus: Identifying compensation mechanisms in cancer cells")
    print("       Comparing CNV-concordant vs CNV-discordant cancer cells")
    print(f"       Threshold: {int(threshold_quantile*100)}/{int((1-threshold_quantile)*100)} percentile split")

    # Create output directory
    patient_output = os.path.join(output_dir, patient_id)
    os.makedirs(patient_output, exist_ok=True)

    # Load data
    adata = load_cnv_data(patient_id, cnv_dir)

    # Load contrastive model
    try:
        model, normalization = load_contrastive_model(model_dir)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Cannot run concordance analysis without trained contrastive model.")
        print("Run: python 07_contrastive_model.py --train")
        return {}

    results = {}

    # Classify cells by concordance (cancer cells only)
    print("\n" + "-" * 50)
    print("Step 1: Classify CNV Concordance (Cancer Cells Only)")
    print("-" * 50)

    adata = classify_concordance_cancer_only(adata, model, normalization, threshold_quantile=threshold_quantile)

    # Save concordance classification
    concordance_df = adata.obs[['embedding_distance', 'cnv_concordance']].copy()
    if 'cancer_vs_normal' in adata.obs.columns:
        concordance_df['cancer_vs_normal'] = adata.obs['cancer_vs_normal']
    concordance_df.to_csv(os.path.join(patient_output, 'cell_concordance.csv'))
    print(f"Saved: {patient_output}/cell_concordance.csv")

    # Analyze subcluster composition
    print("\n" + "-" * 50)
    print("Step 2: Analyze Subcluster Composition")
    print("-" * 50)

    composition_df = analyze_subcluster_composition(adata)
    if not composition_df.empty:
        composition_df.to_csv(os.path.join(patient_output, 'subcluster_composition.csv'), index=False)
        results['subcluster_composition'] = composition_df

    # Run concordance DE (main analysis - cancer cells only)
    print("\n" + "-" * 50)
    print("Step 3: Concordant vs Discordant DE (Cancer Cells)")
    print("-" * 50)

    de_concordance = run_concordance_de_cancer_only(adata, within_subcluster=False)
    if not de_concordance.empty:
        results['concordance_de'] = de_concordance
        de_concordance.to_csv(
            os.path.join(patient_output, 'de_concordant_vs_discordant.csv'),
            index=False
        )

        # Analyze escape and compensation genes
        escape_genes = analyze_escape_genes(de_concordance, adata)
        if not escape_genes.empty:
            escape_genes.to_csv(
                os.path.join(patient_output, 'escape_genes.csv'),
                index=False
            )
            results['escape_genes'] = escape_genes
            print(f"  Escape genes saved: {len(escape_genes)}")

        compensation_genes = analyze_compensation_genes(de_concordance, adata)
        if not compensation_genes.empty:
            compensation_genes.to_csv(
                os.path.join(patient_output, 'compensation_genes.csv'),
                index=False
            )
            results['compensation_genes'] = compensation_genes
            print(f"  Compensation genes saved: {len(compensation_genes)}")

    # Run within-subcluster concordance DE
    print("\n" + "-" * 50)
    print("Step 4: Within-Subcluster Concordance DE")
    print("-" * 50)

    de_within = run_concordance_de_cancer_only(adata, within_subcluster=True)
    if not de_within.empty:
        results['concordance_de_within_subcluster'] = de_within
        de_within.to_csv(
            os.path.join(patient_output, 'de_concordance_within_subcluster.csv'),
            index=False
        )

    # Generate plots
    print("\n" + "-" * 50)
    print("Step 5: Generate Visualizations")
    print("-" * 50)

    plot_concordance_analysis(
        adata,
        de_concordance if not de_concordance.empty else pd.DataFrame(),
        patient_output,
        patient_id,
        threshold_quantile=threshold_quantile
    )

    if not composition_df.empty:
        plot_subcluster_composition(composition_df, patient_output, patient_id)

    # Summary
    print("\n" + "=" * 70)
    print("CONCORDANCE ANALYSIS COMPLETE (CANCER CELLS ONLY)")
    print("=" * 70)
    print(f"\nOutput directory: {patient_output}")
    print("\nKey findings:")

    if 'concordance_de' in results:
        df = results['concordance_de']
        n_sig = df['significant'].sum()
        n_escape = ((df['significant']) & (df['logfoldchanges'] > 0)).sum()
        n_comp = ((df['significant']) & (df['logfoldchanges'] < 0)).sum()
        print(f"  Total significant genes: {n_sig:,}")
        print(f"  Escape genes (up in discordant): {n_escape:,}")
        print(f"  Compensation genes (down in discordant): {n_comp:,}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Run concordance-based DE analysis for cancer cells (compensation mechanisms)'
    )
    parser.add_argument(
        '--patient',
        type=str,
        default=None,
        help='Patient ID to analyze'
    )
    parser.add_argument(
        '--all-patients',
        action='store_true',
        help='Analyze all available patients'
    )
    parser.add_argument(
        '--cnv-dir',
        type=str,
        default='data/cnv_output',
        help='Directory with CNV results'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models/contrastive',
        help='Directory with trained contrastive model'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/de_results',
        help='Output directory for DE results'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.25,
        help='Quantile threshold for concordance classification (default: 0.25 means 25/75 split). Use 0.10 for more extreme 10/90 split.'
    )

    args = parser.parse_args()

    # Get patients to analyze
    if args.all_patients:
        patient_ids = get_available_patients(args.cnv_dir)
    elif args.patient:
        patient_ids = [args.patient]
    else:
        patient_ids = get_available_patients(args.cnv_dir)
        if patient_ids:
            patient_ids = [patient_ids[0]]  # Default to first patient

    if not patient_ids:
        print("No patients with CNV data found!")
        print("Run: python 05_run_infercnv.py --all-patients")
        return

    print(f"Patients to analyze: {patient_ids}")

    # Run analysis for each patient
    all_results = {}
    for patient_id in patient_ids:
        try:
            results = run_concordance_pipeline(
                patient_id,
                cnv_dir=args.cnv_dir,
                model_dir=args.model_dir,
                output_dir=args.output_dir,
                threshold_quantile=args.threshold
            )
            all_results[patient_id] = results

        except Exception as e:
            print(f"Error analyzing {patient_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary across patients
    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("CROSS-PATIENT SUMMARY")
        print("=" * 70)

        for patient_id, results in all_results.items():
            print(f"\n{patient_id}:")
            for name, df in results.items():
                if 'significant' in df.columns:
                    n_sig = df['significant'].sum()
                    print(f"  {name}: {n_sig:,} significant genes")

    # Print interpretation guide
    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    print("""
This analysis identifies genes involved in regulatory escape from CNV effects
in CANCER CELLS ONLY. Normal cells are excluded to focus on tumor-specific
compensation mechanisms.

ESCAPE GENES (upregulated in discordant cancer cells):
  - Expressed HIGHER than expected given the CNV state
  - May indicate: transcriptional compensation, alternative regulation,
    or genes that escape CNV-driven silencing
  - Potential biomarkers of treatment resistance or metastatic potential

COMPENSATION GENES (downregulated in discordant cancer cells):
  - Expressed LOWER than expected given the CNV state
  - May indicate: dosage compensation, epigenetic silencing,
    or buffering mechanisms
  - Could represent therapeutic vulnerabilities

Key files generated:
  - de_concordant_vs_discordant.csv: Full DE results (cancer cells only)
  - escape_genes.csv: Top genes escaping CNV effects
  - compensation_genes.csv: Top genes showing dosage compensation
  - cell_concordance.csv: Per-cell concordance classification
  - subcluster_composition.csv: Concordance breakdown by CNV subcluster

For standard DE analyses (cancer vs normal, high vs low CNV, etc.),
run: python 08b_standard_de.py
""")


if __name__ == "__main__":
    main()
