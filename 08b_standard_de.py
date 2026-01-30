"""
Standard differential expression analyses.

This script contains standard DE comparisons that don't rely on the contrastive model:
- Cancer vs Normal
- High vs Low CNV burden
- CNV subcluster comparisons
- Chromosome-level cis-effects
- CNV differences between normal and cancer

For the main concordance-based analysis using contrastive embeddings,
see 08_differential_expression.py

Usage:
    python 08b_standard_de.py --patient P0006
    python 08b_standard_de.py --all-patients
"""

import os
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy import sparse
from scipy.stats import false_discovery_control
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
    """Run differential expression analysis."""
    group_counts = adata.obs[groupby].value_counts()
    valid_groups = group_counts[group_counts >= min_cells].index.tolist()

    print(f"\n  Group counts for '{groupby}':")
    for g, c in group_counts.items():
        print(f"    {g}: {c:,} cells")

    if groups:
        groups = [g for g in groups if g in valid_groups]
    else:
        groups = valid_groups

    if reference == 'rest':
        if len(groups) < 2:
            print(f"Warning: Not enough groups with >= {min_cells} cells")
            return pd.DataFrame()
    else:
        if len(groups) < 1:
            print(f"Warning: No valid test groups")
            return pd.DataFrame()
        if reference not in valid_groups:
            print(f"Warning: Reference group '{reference}' has < {min_cells} cells")
            return pd.DataFrame()

    print(f"\nRunning DE analysis: {groupby}")
    print(f"  Method: {method}")
    print(f"  Groups: {groups}")
    print(f"  Reference: {reference}")

    sc.tl.rank_genes_groups(
        adata,
        groupby=groupby,
        groups=groups,
        reference=reference,
        method=method,
        pts=True,
        corr_method='benjamini-hochberg'  # Use FDR instead of Bonferroni
    )

    results = []
    for group in groups:
        df = sc.get.rank_genes_groups_df(adata, group=group)
        df['group'] = group
        df['comparison'] = f"{group}_vs_{reference}"
        results.append(df)

    de_results = pd.concat(results, ignore_index=True)

    de_results['significant'] = (
        (de_results['pvals_adj'] < 0.05) &
        (np.abs(de_results['logfoldchanges']) > 0.5)
    )

    n_sig = de_results['significant'].sum()
    print(f"  Significant genes (padj < 0.05, |logFC| > 0.5): {n_sig:,}")

    return de_results


def run_high_vs_low_cnv(adata: sc.AnnData, method: str = 'wilcoxon') -> pd.DataFrame:
    """Run DE between high and low CNV burden cells."""
    if 'cnv_group' not in adata.obs.columns:
        adata = create_cnv_groups(adata, n_groups=2)

    adata_subset = adata[adata.obs['cnv_group'].isin(['High_CNV', 'Low_CNV'])].copy()

    return run_de_analysis(
        adata_subset,
        groupby='cnv_group',
        groups=['High_CNV'],
        reference='Low_CNV',
        method=method
    )


def run_cancer_vs_normal(adata: sc.AnnData, method: str = 'wilcoxon') -> pd.DataFrame:
    """Run DE between cancer and normal cells."""
    return run_de_analysis(
        adata,
        groupby='cancer_vs_normal',
        groups=['Cancer'],
        reference='Normal',
        method=method
    )


def run_subcluster_de(adata: sc.AnnData, top_n_clusters: int = 5, method: str = 'wilcoxon') -> pd.DataFrame:
    """Run DE for top CNV subclusters vs rest."""
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
    """Map chromosomes to their corresponding CNV window indices."""
    if 'cnv' not in adata.uns:
        return {}

    cnv_info = adata.uns.get('cnv', {})
    chr_mapping = {}

    if 'chr_pos' in cnv_info:
        chr_pos = cnv_info['chr_pos']
        n_windows = adata.obsm['X_cnv'].shape[1]
        sorted_chroms = sorted(chr_pos.items(), key=lambda x: x[1])

        for i, (chrom_name, start_idx) in enumerate(sorted_chroms):
            if i + 1 < len(sorted_chroms):
                end_idx = sorted_chroms[i + 1][1]
            else:
                end_idx = n_windows
            chrom = str(chrom_name).replace('chr', '')
            chr_mapping[chrom] = (int(start_idx), int(end_idx))

    return chr_mapping


def analyze_cnv_regions(adata: sc.AnnData, cnv_threshold: float = 0.1) -> pd.DataFrame:
    """Analyze chromosome-level cis-effects."""
    print("\nAnalyzing chromosome-specific cis-effects...")

    cnv_matrix = adata.obsm['X_cnv']
    if hasattr(cnv_matrix, 'toarray'):
        cnv_matrix = cnv_matrix.toarray()

    if 'chromosome' not in adata.var.columns:
        print("Warning: No chromosome info in var")
        return pd.DataFrame()

    chr_cnv_map = get_chromosome_cnv_mapping(adata)
    results = []

    def chrom_sort_key(x):
        x = str(x).replace('chr', '')
        if x == 'X': return 23
        elif x == 'Y': return 24
        try: return int(x)
        except: return 99

    chromosomes = sorted(adata.var['chromosome'].dropna().unique(), key=chrom_sort_key)

    for chrom in chromosomes:
        chrom_genes = adata.var[adata.var['chromosome'] == chrom].index
        if len(chrom_genes) < 10:
            continue

        chrom_expr = adata[:, chrom_genes].X
        if hasattr(chrom_expr, 'toarray'):
            chrom_expr = chrom_expr.toarray()
        mean_expr_per_cell = chrom_expr.mean(axis=1)

        chrom_key = str(chrom).replace('chr', '')
        if chrom_key in chr_cnv_map:
            start_idx, end_idx = chr_cnv_map[chrom_key]
            chrom_cnv_per_cell = cnv_matrix[:, start_idx:end_idx].mean(axis=1)
            cnv_source = 'chromosome-specific'
        else:
            chrom_cnv_per_cell = adata.obs['cnv_score'].values
            cnv_source = 'global-score'

        corr_cis, pval_cis = stats.pearsonr(mean_expr_per_cell, chrom_cnv_per_cell)
        global_cnv = adata.obs['cnv_score'].values
        corr_trans, pval_trans = stats.pearsonr(mean_expr_per_cell, global_cnv)

        results.append({
            'chromosome': chrom,
            'n_genes': len(chrom_genes),
            'cnv_source': cnv_source,
            'cis_correlation': corr_cis,
            'cis_pval': pval_cis,
            'trans_correlation': corr_trans,
            'trans_pval': pval_trans,
            'cis_minus_trans': corr_cis - corr_trans,
        })

    df = pd.DataFrame(results)

    if len(df) > 0:
        df['cis_pval_adj'] = false_discovery_control(df['cis_pval'].values, method='bh')
        df['significant_cis'] = df['cis_pval_adj'] < 0.05

    return df


# ============================================================================
# CNV Differences: Normal vs Cancer
# ============================================================================

def analyze_cnv_normal_vs_cancer(
    adata: sc.AnnData,
    output_dir: str,
    patient_id: str
) -> Dict[str, pd.DataFrame]:
    """Analyze CNV differences between normal and cancer cells."""
    print("\n" + "=" * 60)
    print("CNV Differences: Normal vs Cancer")
    print("=" * 60)

    if 'cancer_vs_normal' not in adata.obs.columns:
        print("Warning: No 'cancer_vs_normal' column found")
        return {}

    cell_counts = adata.obs['cancer_vs_normal'].value_counts()
    print(f"\nCell counts:")
    for group, count in cell_counts.items():
        print(f"  {group}: {count:,}")

    if 'Normal' not in cell_counts.index or 'Cancer' not in cell_counts.index:
        print("Warning: Need both Normal and Cancer cells")
        return {}

    results = {}
    os.makedirs(output_dir, exist_ok=True)

    cnv_matrix = adata.obsm['X_cnv']
    if hasattr(cnv_matrix, 'toarray'):
        cnv_matrix = cnv_matrix.toarray()

    normal_mask = (adata.obs['cancer_vs_normal'] == 'Normal').values
    cancer_mask = (adata.obs['cancer_vs_normal'] == 'Cancer').values

    # 1. Overall CNV burden
    print("\n--- 1. Overall CNV Burden ---")
    cnv_scores_normal = adata.obs.loc[normal_mask, 'cnv_score'].values
    cnv_scores_cancer = adata.obs.loc[cancer_mask, 'cnv_score'].values

    stat, pval = stats.mannwhitneyu(cnv_scores_cancer, cnv_scores_normal, alternative='two-sided')

    burden_df = pd.DataFrame({
        'group': ['Normal', 'Cancer'],
        'n_cells': [normal_mask.sum(), cancer_mask.sum()],
        'mean_cnv_score': [cnv_scores_normal.mean(), cnv_scores_cancer.mean()],
        'median_cnv_score': [np.median(cnv_scores_normal), np.median(cnv_scores_cancer)],
        'std_cnv_score': [cnv_scores_normal.std(), cnv_scores_cancer.std()],
    })
    burden_df['mannwhitney_pval'] = pval

    print(f"  Normal: mean={cnv_scores_normal.mean():.4f}")
    print(f"  Cancer: mean={cnv_scores_cancer.mean():.4f}")
    print(f"  p-value: {pval:.2e}")

    burden_df.to_csv(os.path.join(output_dir, 'cnv_burden_normal_vs_cancer.csv'), index=False)
    results['cnv_burden'] = burden_df

    # 2. Chromosome-level differences
    print("\n--- 2. Chromosome-level CNV Differences ---")
    chr_cnv_map = get_chromosome_cnv_mapping(adata)

    if chr_cnv_map:
        chrom_results = []
        for chrom, (start_idx, end_idx) in chr_cnv_map.items():
            chrom_cnv_normal = cnv_matrix[normal_mask, start_idx:end_idx].mean(axis=1)
            chrom_cnv_cancer = cnv_matrix[cancer_mask, start_idx:end_idx].mean(axis=1)

            stat, pval = stats.mannwhitneyu(chrom_cnv_cancer, chrom_cnv_normal, alternative='two-sided')
            mean_diff = chrom_cnv_cancer.mean() - chrom_cnv_normal.mean()

            chrom_results.append({
                'chromosome': chrom,
                'normal_mean': chrom_cnv_normal.mean(),
                'cancer_mean': chrom_cnv_cancer.mean(),
                'mean_diff': mean_diff,
                'mannwhitney_pval': pval,
            })

        chrom_df = pd.DataFrame(chrom_results)
        chrom_df['pval_adj'] = false_discovery_control(chrom_df['mannwhitney_pval'].values, method='bh')
        chrom_df['significant'] = chrom_df['pval_adj'] < 0.05

        chrom_df.to_csv(os.path.join(output_dir, 'cnv_chromosome_normal_vs_cancer.csv'), index=False)
        results['chromosome_cnv'] = chrom_df

    # 3. Visualization
    _plot_cnv_comparison(adata, cnv_matrix, normal_mask, cancer_mask, output_dir, patient_id)

    return results


def _plot_cnv_comparison(adata, cnv_matrix, normal_mask, cancer_mask, output_dir, patient_id):
    """Generate CNV comparison plots."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    cnv_scores_normal = adata.obs.loc[normal_mask, 'cnv_score'].values
    cnv_scores_cancer = adata.obs.loc[cancer_mask, 'cnv_score'].values

    axes[0].hist(cnv_scores_normal, bins=30, alpha=0.6, label='Normal', color='blue', density=True)
    axes[0].hist(cnv_scores_cancer, bins=30, alpha=0.6, label='Cancer', color='red', density=True)
    axes[0].set_xlabel('CNV Score')
    axes[0].set_ylabel('Density')
    axes[0].set_title(f'{patient_id}: CNV Score Distribution')
    axes[0].legend()

    axes[1].boxplot([cnv_scores_normal, cnv_scores_cancer], labels=['Normal', 'Cancer'])
    axes[1].set_ylabel('CNV Score')
    axes[1].set_title('CNV Burden Comparison')

    parts = axes[2].violinplot([cnv_scores_normal, cnv_scores_cancer], positions=[1, 2], showmeans=True)
    parts['bodies'][0].set_facecolor('blue')
    parts['bodies'][0].set_alpha(0.6)
    parts['bodies'][1].set_facecolor('red')
    parts['bodies'][1].set_alpha(0.6)
    axes[2].set_xticks([1, 2])
    axes[2].set_xticklabels(['Normal', 'Cancer'])
    axes[2].set_ylabel('CNV Score')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cnv_score_normal_vs_cancer.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/cnv_score_normal_vs_cancer.png")


# ============================================================================
# Visualization
# ============================================================================

def plot_volcano(de_results: pd.DataFrame, output_path: str, title: str = 'Volcano Plot'):
    """Create volcano plot for DE results."""
    fig, ax = plt.subplots(figsize=(10, 8))

    de_results = de_results.copy()
    de_results['-log10_pval'] = -np.log10(de_results['pvals_adj'] + 1e-300)

    colors = []
    for _, row in de_results.iterrows():
        if row['pvals_adj'] < 0.05:
            if row['logfoldchanges'] > 0.5:
                colors.append('red')
            elif row['logfoldchanges'] < -0.5:
                colors.append('blue')
            else:
                colors.append('gray')
        else:
            colors.append('gray')

    ax.scatter(de_results['logfoldchanges'], de_results['-log10_pval'], c=colors, alpha=0.5, s=10)
    ax.axhline(-np.log10(0.05), color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(-0.5, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel('Log2 Fold Change')
    ax.set_ylabel('-Log10 Adjusted P-value')
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================================
# Main Pipeline
# ============================================================================

def run_standard_de_pipeline(
    patient_id: str,
    cnv_dir: str = 'data/cnv_output',
    output_dir: str = 'data/de_results'
) -> Dict[str, pd.DataFrame]:
    """Run standard DE analysis pipeline."""
    print("=" * 60)
    print(f"Standard DE Analysis: {patient_id}")
    print("=" * 60)

    patient_output = os.path.join(output_dir, patient_id)
    os.makedirs(patient_output, exist_ok=True)

    adata = load_cnv_data(patient_id, cnv_dir)
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
        plot_volcano(de_cnv, os.path.join(patient_output, 'volcano_high_vs_low_cnv.png'),
                    title=f'{patient_id}: High vs Low CNV')

    # 2. Cancer vs Normal
    print("\n" + "-" * 40)
    print("Analysis 2: Cancer vs Normal")
    print("-" * 40)

    if 'Normal' in adata.obs['cancer_vs_normal'].values:
        de_cancer = run_cancer_vs_normal(adata)
        if not de_cancer.empty:
            results['cancer_vs_normal'] = de_cancer
            de_cancer.to_csv(os.path.join(patient_output, 'de_cancer_vs_normal.csv'), index=False)
            plot_volcano(de_cancer, os.path.join(patient_output, 'volcano_cancer_vs_normal.png'),
                        title=f'{patient_id}: Cancer vs Normal')

        # CNV differences
        cnv_diff_results = analyze_cnv_normal_vs_cancer(adata, patient_output, patient_id)
        results.update(cnv_diff_results)
    else:
        print("Skipping: No normal cells")

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

    # Summary
    print("\n" + "=" * 60)
    print("STANDARD DE ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {patient_output}")

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
    parser = argparse.ArgumentParser(description='Standard differential expression analyses')
    parser.add_argument('--patient', type=str, default=None, help='Patient ID')
    parser.add_argument('--all-patients', action='store_true', help='Analyze all patients')
    parser.add_argument('--cnv-dir', type=str, default='data/cnv_output')
    parser.add_argument('--output-dir', type=str, default='data/de_results')

    args = parser.parse_args()

    if args.all_patients:
        patient_ids = get_available_patients(args.cnv_dir)
    elif args.patient:
        patient_ids = [args.patient]
    else:
        patient_ids = get_available_patients(args.cnv_dir)
        if patient_ids:
            patient_ids = [patient_ids[0]]

    if not patient_ids:
        print("No patients found!")
        return

    for patient_id in patient_ids:
        try:
            run_standard_de_pipeline(patient_id, args.cnv_dir, args.output_dir)
        except Exception as e:
            print(f"Error analyzing {patient_id}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
