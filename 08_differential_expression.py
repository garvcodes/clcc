"""
Differential expression analysis based on CNV status.

This script identifies genes differentially expressed between:
1. High-CNV vs Low-CNV cells (based on cnv_score)
2. Cancer vs Normal cells
3. Between CNV subclusters
4. Genes in amplified/deleted chromosomal regions

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
from scipy import stats
from typing import Dict, List, Tuple, Optional
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


def load_all_patients(
    patient_ids: List[str],
    cnv_dir: str = 'data/cnv_output'
) -> sc.AnnData:
    """Load and concatenate data from multiple patients."""
    adatas = []

    for patient_id in patient_ids:
        try:
            adata = load_cnv_data(patient_id, cnv_dir)
            adata.obs['patient_id'] = patient_id
            # Make CNV leiden unique per patient
            adata.obs['cnv_leiden_patient'] = patient_id + '_' + adata.obs['cnv_leiden'].astype(str)
            adatas.append(adata)
        except Exception as e:
            print(f"Warning: Could not load {patient_id}: {e}")
            continue

    if not adatas:
        raise ValueError("No patient data could be loaded")

    # Concatenate
    adata = sc.concat(adatas, join='inner')
    print(f"\nCombined: {adata.n_obs:,} cells, {adata.n_vars:,} genes")

    return adata


# ============================================================================
# CNV Score Groups
# ============================================================================

def create_cnv_groups(
    adata: sc.AnnData,
    method: str = 'quantile',
    n_groups: int = 3,
    quantiles: Tuple[float, float] = (0.25, 0.75)
) -> sc.AnnData:
    """
    Create CNV burden groups based on cnv_score.

    Args:
        adata: AnnData with cnv_score in obs
        method: 'quantile' or 'fixed'
        n_groups: Number of groups (2 or 3)
        quantiles: Quantile thresholds for 3-group split

    Returns:
        AnnData with 'cnv_group' column added
    """
    scores = adata.obs['cnv_score'].values

    if method == 'quantile':
        if n_groups == 2:
            median = np.median(scores)
            groups = np.where(scores >= median, 'High_CNV', 'Low_CNV')
        else:
            q_low, q_high = np.quantile(scores, quantiles)
            groups = np.where(scores <= q_low, 'Low_CNV',
                            np.where(scores >= q_high, 'High_CNV', 'Mid_CNV'))
    else:
        # Fixed thresholds based on score distribution
        threshold = scores.mean() + scores.std()
        groups = np.where(scores >= threshold, 'High_CNV', 'Low_CNV')

    adata.obs['cnv_group'] = pd.Categorical(groups)

    print(f"\nCNV groups:")
    for group in adata.obs['cnv_group'].cat.categories:
        n = (adata.obs['cnv_group'] == group).sum()
        print(f"  {group}: {n:,} cells")

    return adata


# ============================================================================
# Differential Expression
# ============================================================================

def run_de_analysis(
    adata: sc.AnnData,
    groupby: str,
    groups: Optional[List[str]] = None,
    reference: str = 'rest',
    method: str = 'wilcoxon',
    min_cells: int = 10
) -> pd.DataFrame:
    """
    Run differential expression analysis.

    Args:
        adata: AnnData object
        groupby: Column in obs to group by
        groups: Specific groups to test (default: all)
        reference: Reference group ('rest' or specific group name)
        method: DE method ('wilcoxon', 't-test', 't-test_overestim_var')
        min_cells: Minimum cells per group

    Returns:
        DataFrame with DE results
    """
    # Filter groups with too few cells
    group_counts = adata.obs[groupby].value_counts()
    valid_groups = group_counts[group_counts >= min_cells].index.tolist()

    # Debug: show what we have
    print(f"\n  Group counts for '{groupby}':")
    for g, c in group_counts.items():
        print(f"    {g}: {c:,} cells")

    if groups:
        groups = [g for g in groups if g in valid_groups]
    else:
        groups = valid_groups

    # Check if we have enough groups
    # For 'rest' reference: need at least 2 valid groups
    # For specific reference: need at least 1 group AND reference must be valid
    if reference == 'rest':
        if len(groups) < 2:
            print(f"Warning: Not enough groups with >= {min_cells} cells for 'rest' comparison")
            return pd.DataFrame()
    else:
        # Specific reference - check if both group and reference are valid
        if len(groups) < 1:
            print(f"Warning: No valid test groups with >= {min_cells} cells")
            return pd.DataFrame()
        if reference not in valid_groups:
            print(f"Warning: Reference group '{reference}' has < {min_cells} cells")
            return pd.DataFrame()

    print(f"\nRunning DE analysis: {groupby}")
    print(f"  Method: {method}")
    print(f"  Groups: {groups}")
    print(f"  Reference: {reference}")

    # Run DE
    sc.tl.rank_genes_groups(
        adata,
        groupby=groupby,
        groups=groups,
        reference=reference,
        method=method,
        pts=True  # Include percentage of cells expressing
    )

    # Extract results
    results = []
    for group in groups:
        df = sc.get.rank_genes_groups_df(adata, group=group)
        df['group'] = group
        df['comparison'] = f"{group}_vs_{reference}"
        results.append(df)

    de_results = pd.concat(results, ignore_index=True)

    # Add significance flags
    de_results['significant'] = (
        (de_results['pvals_adj'] < 0.05) &
        (np.abs(de_results['logfoldchanges']) > 0.5)
    )

    n_sig = de_results['significant'].sum()
    print(f"  Significant genes (padj < 0.05, |logFC| > 0.5): {n_sig:,}")

    return de_results


def run_high_vs_low_cnv(
    adata: sc.AnnData,
    method: str = 'wilcoxon'
) -> pd.DataFrame:
    """Run DE between high and low CNV burden cells."""
    # Create CNV groups if not present
    if 'cnv_group' not in adata.obs.columns:
        adata = create_cnv_groups(adata, n_groups=2)

    # Filter to just High and Low
    adata_subset = adata[adata.obs['cnv_group'].isin(['High_CNV', 'Low_CNV'])].copy()

    return run_de_analysis(
        adata_subset,
        groupby='cnv_group',
        groups=['High_CNV'],
        reference='Low_CNV',
        method=method
    )


def run_cancer_vs_normal(
    adata: sc.AnnData,
    method: str = 'wilcoxon'
) -> pd.DataFrame:
    """Run DE between cancer and normal cells."""
    return run_de_analysis(
        adata,
        groupby='cancer_vs_normal',
        groups=['Cancer'],
        reference='Normal',
        method=method
    )


def run_subcluster_de(
    adata: sc.AnnData,
    top_n_clusters: int = 5,
    method: str = 'wilcoxon'
) -> pd.DataFrame:
    """Run DE for top CNV subclusters vs rest."""
    # Find top clusters by CNV score
    cluster_scores = adata.obs.groupby('cnv_leiden')['cnv_score'].mean()
    top_clusters = cluster_scores.nlargest(top_n_clusters).index.tolist()

    print(f"\nTop {top_n_clusters} CNV subclusters by mean score:")
    for c in top_clusters:
        score = cluster_scores[c]
        n = (adata.obs['cnv_leiden'] == c).sum()
        print(f"  Cluster {c}: score={score:.3f}, n={n:,}")

    return run_de_analysis(
        adata,
        groupby='cnv_leiden',
        groups=[str(c) for c in top_clusters],
        reference='rest',
        method=method
    )


# ============================================================================
# Chromosome-specific Analysis
# ============================================================================

def get_chromosome_cnv_mapping(adata: sc.AnnData) -> Dict[str, Tuple[int, int]]:
    """
    Map chromosomes to their corresponding CNV window indices.

    The X_cnv matrix columns are genomic windows. We need to figure out
    which windows correspond to which chromosomes based on the uns metadata.

    infercnvpy stores this in adata.uns['cnv']['chr_pos'] as a dict mapping
    chromosome names (e.g., 'chr1') to their START index in the X_cnv matrix.

    Returns:
        Dict mapping chromosome (without 'chr' prefix) to (start_idx, end_idx) tuple
    """
    # infercnvpy stores window info in uns
    if 'cnv' not in adata.uns:
        print("  Note: No 'cnv' key in adata.uns")
        return {}

    cnv_info = adata.uns.get('cnv', {})
    print(f"  CNV uns keys: {list(cnv_info.keys())}")

    chr_mapping = {}

    # infercnvpy stores chr_pos: dict mapping 'chr1' -> start_index
    if 'chr_pos' in cnv_info:
        chr_pos = cnv_info['chr_pos']
        print(f"  Found 'chr_pos' with {len(chr_pos)} chromosomes")

        # Get total number of windows
        n_windows = adata.obsm['X_cnv'].shape[1]

        # Sort by start index to determine end indices
        sorted_chroms = sorted(chr_pos.items(), key=lambda x: x[1])

        for i, (chrom_name, start_idx) in enumerate(sorted_chroms):
            # End index is the start of the next chromosome, or n_windows
            if i + 1 < len(sorted_chroms):
                end_idx = sorted_chroms[i + 1][1]
            else:
                end_idx = n_windows

            # Normalize chromosome name (remove 'chr' prefix if present)
            chrom = str(chrom_name).replace('chr', '')
            chr_mapping[chrom] = (int(start_idx), int(end_idx))

        print(f"  Mapped {len(chr_mapping)} chromosomes to CNV window ranges")

    elif 'chr' in cnv_info:
        # Alternative format: array of chromosome labels per window
        chr_labels = cnv_info['chr']
        print(f"  Found 'chr' array with {len(chr_labels)} entries")

        # Group consecutive windows by chromosome
        current_chr = None
        start_idx = 0
        for i, chrom in enumerate(chr_labels):
            chrom = str(chrom).replace('chr', '')
            if chrom != current_chr:
                if current_chr is not None:
                    chr_mapping[current_chr] = (start_idx, i)
                current_chr = chrom
                start_idx = i
        # Don't forget the last chromosome
        if current_chr is not None:
            chr_mapping[current_chr] = (start_idx, len(chr_labels))

        print(f"  Mapped {len(chr_mapping)} chromosomes to CNV window ranges")
    else:
        print("  Note: No chromosome info found in cnv uns - will use global CNV score")

    return chr_mapping


def analyze_cnv_regions(
    adata: sc.AnnData,
    cnv_threshold: float = 0.1
) -> pd.DataFrame:
    """
    Identify chromosomal regions with significant CNV and test for
    TRUE cis-effects: correlation between chromosome-specific CNV and
    expression of genes ON THAT SAME chromosome.

    This tests the gene dosage hypothesis: if a chromosome is amplified,
    are genes on that chromosome upregulated proportionally?

    Args:
        adata: AnnData with CNV data
        cnv_threshold: Threshold for calling amplification/deletion

    Returns:
        DataFrame with per-chromosome cis-effect analysis
    """
    print("\nAnalyzing chromosome-specific cis-effects...")
    print("Testing: Does chr X CNV correlate with chr X gene expression?")

    cnv_matrix = adata.obsm['X_cnv']
    if hasattr(cnv_matrix, 'toarray'):
        cnv_matrix = cnv_matrix.toarray()

    if 'chromosome' not in adata.var.columns:
        print("Warning: No chromosome info in var, skipping region analysis")
        return pd.DataFrame()

    # Try to get chromosome-to-CNV-window mapping
    chr_cnv_map = get_chromosome_cnv_mapping(adata)

    results = []
    chromosomes = sorted(adata.var['chromosome'].dropna().unique())

    # Sort chromosomes properly
    def chrom_sort_key(x):
        x = str(x).replace('chr', '')
        if x == 'X':
            return 23
        elif x == 'Y':
            return 24
        else:
            try:
                return int(x)
            except:
                return 99

    chromosomes = sorted(chromosomes, key=chrom_sort_key)

    for chrom in chromosomes:
        # Get genes on this chromosome
        chrom_genes = adata.var[adata.var['chromosome'] == chrom].index

        if len(chrom_genes) < 10:
            continue

        # Get mean expression of genes on this chromosome (per cell)
        chrom_expr = adata[:, chrom_genes].X
        if hasattr(chrom_expr, 'toarray'):
            chrom_expr = chrom_expr.toarray()
        mean_expr_per_cell = chrom_expr.mean(axis=1)

        # Get CNV for this specific chromosome
        # Normalize chromosome name to match mapping (without 'chr' prefix)
        chrom_key = str(chrom).replace('chr', '')
        if chrom_key in chr_cnv_map:
            # Use chromosome-specific CNV windows
            start_idx, end_idx = chr_cnv_map[chrom_key]
            chrom_cnv_per_cell = cnv_matrix[:, start_idx:end_idx].mean(axis=1)
            cnv_source = 'chromosome-specific'
        else:
            # Fallback: estimate from expression-derived CNV
            # This is less accurate but still informative
            chrom_cnv_per_cell = adata.obs['cnv_score'].values
            cnv_source = 'global-score'

        # TRUE CIS-EFFECT: Correlate chr X expression with chr X CNV
        corr_cis, pval_cis = stats.pearsonr(mean_expr_per_cell, chrom_cnv_per_cell)

        # TRANS-EFFECT comparison: Correlate chr X expression with global CNV
        # (only meaningful if we have chromosome-specific CNV)
        global_cnv = adata.obs['cnv_score'].values
        corr_trans, pval_trans = stats.pearsonr(mean_expr_per_cell, global_cnv)

        # Calculate mean CNV for this chromosome
        mean_chrom_cnv = chrom_cnv_per_cell.mean()
        std_chrom_cnv = chrom_cnv_per_cell.std()

        # Identify cells with amplification vs deletion for this chromosome
        amp_cells = chrom_cnv_per_cell > (mean_chrom_cnv + std_chrom_cnv)
        del_cells = chrom_cnv_per_cell < (mean_chrom_cnv - std_chrom_cnv)

        expr_amp = mean_expr_per_cell[amp_cells].mean() if amp_cells.sum() > 5 else np.nan
        expr_del = mean_expr_per_cell[del_cells].mean() if del_cells.sum() > 5 else np.nan
        expr_neutral = mean_expr_per_cell[~amp_cells & ~del_cells].mean()

        # Fold changes
        fc_amp_vs_neutral = expr_amp / (expr_neutral + 1e-8) if not np.isnan(expr_amp) else np.nan
        fc_del_vs_neutral = expr_del / (expr_neutral + 1e-8) if not np.isnan(expr_del) else np.nan

        results.append({
            'chromosome': chrom,
            'n_genes': len(chrom_genes),
            'cnv_source': cnv_source,
            'mean_cnv': mean_chrom_cnv,
            'std_cnv': std_chrom_cnv,
            # Cis-effect (the key metric)
            'cis_correlation': corr_cis,
            'cis_pval': pval_cis,
            # Trans-effect for comparison
            'trans_correlation': corr_trans,
            'trans_pval': pval_trans,
            # Cis > Trans suggests true dosage effect
            'cis_minus_trans': corr_cis - corr_trans,
            # Expression in different CNV states
            'n_amplified': amp_cells.sum(),
            'n_deleted': del_cells.sum(),
            'expr_amplified': expr_amp,
            'expr_deleted': expr_del,
            'expr_neutral': expr_neutral,
            'fc_amp_vs_neutral': fc_amp_vs_neutral,
            'fc_del_vs_neutral': fc_del_vs_neutral,
        })

    df = pd.DataFrame(results)

    # Significance: cis-correlation with proper multiple testing correction
    from scipy.stats import false_discovery_control
    if len(df) > 0 and 'cis_pval' in df.columns:
        # FDR correction
        df['cis_pval_adj'] = false_discovery_control(df['cis_pval'].values, method='bh')
        df['significant_cis'] = df['cis_pval_adj'] < 0.05

        # Strong cis-effect: significant AND positive correlation AND cis > trans
        df['strong_cis_effect'] = (
            (df['cis_pval_adj'] < 0.05) &
            (df['cis_correlation'] > 0.1) &
            (df['cis_minus_trans'] > 0)
        )

    n_sig = df['significant_cis'].sum() if 'significant_cis' in df.columns else 0
    n_strong = df['strong_cis_effect'].sum() if 'strong_cis_effect' in df.columns else 0

    print(f"\nResults:")
    print(f"  Chromosomes with significant cis-effects: {n_sig}")
    print(f"  Chromosomes with STRONG cis-effects (cis > trans): {n_strong}")

    return df


def analyze_gene_level_cis_effects(
    adata: sc.AnnData,
    top_n: int = 100
) -> pd.DataFrame:
    """
    Analyze TRUE cis-effects at the individual gene level.

    For each gene, correlate its expression with the CNV of ITS OWN chromosome,
    not the global CNV score. This tests whether gene expression tracks
    local copy number changes.

    Genes with high positive cis-correlation are "dosage-sensitive".
    Genes with no cis-correlation despite CNV changes are "dosage-compensated".

    Returns:
        DataFrame with per-gene cis-effect analysis
    """
    print("\nAnalyzing gene-level cis-effects...")

    if 'chromosome' not in adata.var.columns:
        print("Warning: No chromosome info, skipping gene-level analysis")
        return pd.DataFrame()

    # Get chromosome to CNV window mapping
    chr_cnv_map = get_chromosome_cnv_mapping(adata)

    # Get CNV matrix
    cnv_matrix = adata.obsm['X_cnv']
    if hasattr(cnv_matrix, 'toarray'):
        cnv_matrix = cnv_matrix.toarray()

    global_cnv = adata.obs['cnv_score'].values

    results = []

    # Get expression matrix
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()

    # Pre-compute per-chromosome CNV scores for efficiency
    chrom_cnv_scores = {}
    for chrom_key, (start_idx, end_idx) in chr_cnv_map.items():
        chrom_cnv_scores[chrom_key] = cnv_matrix[:, start_idx:end_idx].mean(axis=1)

    for i, gene in enumerate(adata.var_names):
        gene_expr = X[:, i]
        chrom = adata.var.loc[gene, 'chromosome']

        # Skip genes with no variation
        if gene_expr.std() < 1e-6:
            continue

        # Normalize chromosome name
        chrom_key = str(chrom).replace('chr', '') if pd.notna(chrom) else None

        # Get chromosome-specific CNV if available
        if chrom_key and chrom_key in chrom_cnv_scores:
            chrom_cnv = chrom_cnv_scores[chrom_key]
            # TRUE CIS-EFFECT: correlate gene expression with its chromosome's CNV
            corr_cis, pval_cis = stats.pearsonr(gene_expr, chrom_cnv)
            # TRANS-EFFECT: correlate with global CNV for comparison
            corr_trans, pval_trans = stats.pearsonr(gene_expr, global_cnv)
            cnv_source = 'chromosome-specific'
        else:
            # Fallback to global CNV
            corr_cis, pval_cis = stats.pearsonr(gene_expr, global_cnv)
            corr_trans, pval_trans = corr_cis, pval_cis  # Same as cis
            cnv_source = 'global'

        results.append({
            'gene': gene,
            'chromosome': chrom,
            'mean_expr': gene_expr.mean(),
            'std_expr': gene_expr.std(),
            'corr_cis': corr_cis,  # Correlation with own chromosome CNV
            'pval_cis': pval_cis,
            'corr_trans': corr_trans,  # Correlation with global CNV
            'pval_trans': pval_trans,
            'cis_minus_trans': corr_cis - corr_trans,  # Positive = true cis-effect
            'cnv_source': cnv_source,
            # Keep old name for backwards compatibility
            'corr_with_cnv': corr_cis,
            'pval': pval_cis
        })

    df = pd.DataFrame(results)

    if len(df) > 0:
        # FDR correction on cis p-values
        from scipy.stats import false_discovery_control
        df['pval_adj'] = false_discovery_control(df['pval_cis'].values, method='bh')

        # Classify genes based on CIS correlation (not global)
        # Dosage-sensitive: significant positive cis-correlation
        df['dosage_sensitive'] = (df['pval_adj'] < 0.05) & (df['corr_cis'] > 0.2)

        # Dosage-compensated: no correlation despite being in CNV region
        df['dosage_compensated'] = (df['pval_adj'] > 0.1) & (df['corr_cis'].abs() < 0.1)

        # Strong cis-effect: cis correlation significantly exceeds trans
        df['strong_cis_effect'] = (
            (df['pval_adj'] < 0.05) &
            (df['corr_cis'] > 0.15) &
            (df['cis_minus_trans'] > 0.05)
        )

        # Sort by cis correlation
        df = df.sort_values('corr_cis', ascending=False)

        n_sensitive = df['dosage_sensitive'].sum()
        n_compensated = df['dosage_compensated'].sum()
        n_strong_cis = df['strong_cis_effect'].sum()
        n_chr_specific = (df['cnv_source'] == 'chromosome-specific').sum()

        print(f"  Genes with chromosome-specific CNV: {n_chr_specific:,}")
        print(f"  Dosage-sensitive genes: {n_sensitive:,}")
        print(f"  Dosage-compensated genes: {n_compensated:,}")
        print(f"  Strong cis-effect genes (cis >> trans): {n_strong_cis:,}")

    return df


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
        Patch(facecolor='red', label=f'Up (FC>{fc_threshold}, p<{pval_threshold})'),
        Patch(facecolor='blue', label=f'Down (FC<-{fc_threshold}, p<{pval_threshold})'),
        Patch(facecolor='gray', label='Not significant')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def plot_top_genes_heatmap(
    adata: sc.AnnData,
    de_results: pd.DataFrame,
    groupby: str,
    output_path: str,
    top_n: int = 50,
    title: str = 'Top DE Genes'
):
    """Plot heatmap of top DE genes."""
    # Get top genes by significance
    top_genes = de_results.nlargest(top_n, 'scores')['names'].values

    # Subset to these genes
    adata_subset = adata[:, [g for g in top_genes if g in adata.var_names]].copy()

    # Plot
    sc.pl.heatmap(
        adata_subset,
        var_names=adata_subset.var_names.tolist(),
        groupby=groupby,
        show=False,
        save=False,
        figsize=(12, 10)
    )

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def plot_chromosome_effects(
    chrom_results: pd.DataFrame,
    output_path: str
):
    """Plot chromosome-level cis-effect analysis."""
    if chrom_results.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Sort chromosomes
    def chrom_sort_key(x):
        x = str(x).replace('chr', '')
        if x == 'X':
            return 23
        elif x == 'Y':
            return 24
        else:
            try:
                return int(x)
            except:
                return 99

    chrom_results = chrom_results.sort_values('chromosome', key=lambda x: x.map(chrom_sort_key))

    # Plot 1: Cis-correlation (chr X expression vs chr X CNV)
    if 'cis_correlation' in chrom_results.columns:
        colors = ['red' if (p < 0.05 and c > 0) else 'blue' if (p < 0.05 and c < 0) else 'gray'
                  for p, c in zip(chrom_results['cis_pval'], chrom_results['cis_correlation'])]
        axes[0].bar(range(len(chrom_results)), chrom_results['cis_correlation'], color=colors)
        axes[0].set_ylabel('Cis-Correlation (Chr Expression vs Chr CNV)')
        axes[0].set_title('Cis-Effects: Same-Chromosome Expression-CNV Correlation')
    else:
        # Fallback to old format
        colors = ['red' if p < 0.05 else 'gray' for p in chrom_results.get('corr_pval', [0.1]*len(chrom_results))]
        axes[0].bar(range(len(chrom_results)), chrom_results.get('corr_expr_cnv', [0]*len(chrom_results)), color=colors)
        axes[0].set_ylabel('Correlation (Expression vs CNV)')
        axes[0].set_title('Expression-CNV Correlation by Chromosome')

    axes[0].set_xticks(range(len(chrom_results)))
    axes[0].set_xticklabels(chrom_results['chromosome'], rotation=45)
    axes[0].set_xlabel('Chromosome')
    axes[0].axhline(0, color='black', linestyle='-', alpha=0.3)

    # Plot 2: Cis vs Trans comparison OR fold change
    if 'cis_minus_trans' in chrom_results.columns:
        # Show cis - trans difference (positive = true cis-effect)
        colors = ['green' if d > 0.05 else 'gray' for d in chrom_results['cis_minus_trans']]
        axes[1].bar(range(len(chrom_results)), chrom_results['cis_minus_trans'], color=colors)
        axes[1].set_ylabel('Cis - Trans Correlation')
        axes[1].set_title('Cis-Specificity (Positive = True Dosage Effect)')
    elif 'fc_amp_vs_neutral' in chrom_results.columns:
        # Show fold change for amplified regions
        fc_values = chrom_results['fc_amp_vs_neutral'].fillna(1)
        colors = ['red' if fc > 1.1 else 'blue' if fc < 0.9 else 'gray' for fc in fc_values]
        axes[1].bar(range(len(chrom_results)), np.log2(fc_values), color=colors)
        axes[1].set_ylabel('Log2 FC (Amplified vs Neutral)')
        axes[1].set_title('Expression in Amplified Regions')
    else:
        # Fallback
        fc_values = chrom_results.get('fold_change', [1]*len(chrom_results))
        colors = ['red' if fc > 1.1 else 'blue' if fc < 0.9 else 'gray' for fc in fc_values]
        axes[1].bar(range(len(chrom_results)), np.log2(fc_values), color=colors)
        axes[1].set_ylabel('Log2 Fold Change')
        axes[1].set_title('Expression Fold Change by Chromosome')

    axes[1].set_xticks(range(len(chrom_results)))
    axes[1].set_xticklabels(chrom_results['chromosome'], rotation=45)
    axes[1].set_xlabel('Chromosome')
    axes[1].axhline(0, color='black', linestyle='-', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


# ============================================================================
# Main Pipeline
# ============================================================================

def run_de_pipeline(
    patient_id: str,
    cnv_dir: str = 'data/cnv_output',
    output_dir: str = 'data/de_results'
) -> Dict[str, pd.DataFrame]:
    """
    Run complete DE analysis pipeline for a patient.

    Returns:
        Dictionary of DE result DataFrames
    """
    print("=" * 60)
    print(f"Differential Expression Analysis: {patient_id}")
    print("=" * 60)

    # Create output directory
    patient_output = os.path.join(output_dir, patient_id)
    os.makedirs(patient_output, exist_ok=True)

    # Load data
    adata = load_cnv_data(patient_id, cnv_dir)

    # Create CNV groups
    adata = create_cnv_groups(adata, n_groups=2)

    results = {}

    # 1. High vs Low CNV
    print("\n" + "-" * 40)
    print("Analysis 1: High vs Low CNV")
    print("-" * 40)

    de_cnv = run_high_vs_low_cnv(adata)
    if not de_cnv.empty:
        results['high_vs_low_cnv'] = de_cnv
        de_cnv.to_csv(os.path.join(patient_output, 'de_high_vs_low_cnv.csv'), index=False)

        plot_volcano(
            de_cnv,
            os.path.join(patient_output, 'volcano_high_vs_low_cnv.png'),
            title=f'{patient_id}: High vs Low CNV'
        )

    # 2. Cancer vs Normal
    print("\n" + "-" * 40)
    print("Analysis 2: Cancer vs Normal")
    print("-" * 40)

    if 'Normal' in adata.obs['cancer_vs_normal'].values:
        de_cancer = run_cancer_vs_normal(adata)
        if not de_cancer.empty:
            results['cancer_vs_normal'] = de_cancer
            de_cancer.to_csv(os.path.join(patient_output, 'de_cancer_vs_normal.csv'), index=False)

            plot_volcano(
                de_cancer,
                os.path.join(patient_output, 'volcano_cancer_vs_normal.png'),
                title=f'{patient_id}: Cancer vs Normal'
            )
    else:
        print("Skipping: No normal cells in dataset")

    # 3. Top CNV subclusters
    print("\n" + "-" * 40)
    print("Analysis 3: Top CNV Subclusters")
    print("-" * 40)

    de_clusters = run_subcluster_de(adata, top_n_clusters=3)
    if not de_clusters.empty:
        results['cnv_subclusters'] = de_clusters
        de_clusters.to_csv(os.path.join(patient_output, 'de_cnv_subclusters.csv'), index=False)

    # 4. Chromosome-level cis-effects
    print("\n" + "-" * 40)
    print("Analysis 4: Chromosome-level cis-effects")
    print("-" * 40)

    chrom_results = analyze_cnv_regions(adata)
    if not chrom_results.empty:
        results['chromosome_cis_effects'] = chrom_results
        chrom_results.to_csv(os.path.join(patient_output, 'chromosome_cis_effects.csv'), index=False)

        plot_chromosome_effects(
            chrom_results,
            os.path.join(patient_output, 'chromosome_cis_effects.png')
        )

    # 5. Gene-level cis-effects (dosage sensitivity)
    print("\n" + "-" * 40)
    print("Analysis 5: Gene-level dosage sensitivity")
    print("-" * 40)

    gene_cis = analyze_gene_level_cis_effects(adata)
    if not gene_cis.empty:
        results['gene_cis_effects'] = gene_cis
        gene_cis.to_csv(os.path.join(patient_output, 'gene_dosage_sensitivity.csv'), index=False)

        # Save top dosage-sensitive genes separately
        sensitive = gene_cis[gene_cis['dosage_sensitive'] == True]
        if len(sensitive) > 0:
            sensitive.to_csv(os.path.join(patient_output, 'dosage_sensitive_genes.csv'), index=False)
            print(f"  Top dosage-sensitive genes saved")

    # Summary
    print("\n" + "=" * 60)
    print("DE ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {patient_output}")
    print("\nFiles generated:")
    for name, df in results.items():
        n_sig = df['significant'].sum() if 'significant' in df.columns else len(df)
        print(f"  {name}: {n_sig:,} significant")

    return results


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
        description='Run differential expression analysis based on CNV status'
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
        '--output-dir',
        type=str,
        default='data/de_results',
        help='Output directory for DE results'
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
            results = run_de_pipeline(
                patient_id,
                cnv_dir=args.cnv_dir,
                output_dir=args.output_dir
            )
            all_results[patient_id] = results
        except Exception as e:
            print(f"Error analyzing {patient_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary across patients
    if len(all_results) > 1:
        print("\n" + "=" * 60)
        print("CROSS-PATIENT SUMMARY")
        print("=" * 60)

        for patient_id, results in all_results.items():
            print(f"\n{patient_id}:")
            for name, df in results.items():
                if 'significant' in df.columns:
                    n_sig = df['significant'].sum()
                    print(f"  {name}: {n_sig:,} significant genes")


if __name__ == "__main__":
    main()
