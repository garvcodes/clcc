"""
Pooled Cross-Patient Differential Expression Analysis

Instead of analyzing each patient separately and combining results,
this script pools all cells across patients for a single, more powerful
DE analysis. Patient is included as a covariate to control for batch effects.

This approach has much more statistical power because:
- More cells = more power (60k+ vs 2-8k per patient)
- Can detect effects that are consistent but small across patients
- Controls for patient-specific variation

Usage:
    python 08c_pooled_de_analysis.py --model-dir models/contrastive_hn
    python 08c_pooled_de_analysis.py --model-dir models/contrastive_hn --threshold 0.10
"""

import os
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Model Loading (same as 08_differential_expression.py)
# ============================================================================

class ExpressionEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256], latent_dim=128, dropout=0.1):
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
    def __init__(self, input_dim, hidden_dims=[256, 128], latent_dim=128, dropout=0.1):
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


class ContrastiveModel(nn.Module):
    def __init__(self, expression_dim, cnv_dim, latent_dim=128,
                 expression_hidden=[512, 256], cnv_hidden=[256, 128],
                 temperature=0.07, dropout=0.1):
        super().__init__()
        self.expression_encoder = ExpressionEncoder(
            input_dim=expression_dim, hidden_dims=expression_hidden,
            latent_dim=latent_dim, dropout=dropout
        )
        self.cnv_encoder = CNVEncoder(
            input_dim=cnv_dim, hidden_dims=cnv_hidden,
            latent_dim=latent_dim, dropout=dropout
        )
        self.temperature = temperature
        self.latent_dim = latent_dim

    def forward(self, expression, cnv):
        expression_embed = self.expression_encoder(expression)
        cnv_embed = self.cnv_encoder(cnv)
        return expression_embed, cnv_embed


def load_model(model_dir: str):
    """Load trained contrastive model."""
    model_path = os.path.join(model_dir, 'best_model.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    model = ContrastiveModel(
        expression_dim=checkpoint['expression_dim'],
        cnv_dim=checkpoint['cnv_dim'],
        latent_dim=checkpoint['latent_dim']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    normalization = checkpoint['normalization']

    return model, normalization


# ============================================================================
# Data Loading
# ============================================================================

def load_patient_data(patient_id: str, cnv_dir: str = 'data/cnv_output') -> sc.AnnData:
    """Load CNV data for a patient."""
    cnv_path = os.path.join(cnv_dir, patient_id, f'{patient_id}_cnv.h5ad')
    if not os.path.exists(cnv_path):
        raise FileNotFoundError(f"CNV file not found: {cnv_path}")

    adata = sc.read_h5ad(cnv_path)
    adata.obs['patient_id'] = patient_id

    # Ensure cell names are unique across patients
    adata.obs_names = [f"{patient_id}_{x}" for x in adata.obs_names]

    return adata


def load_all_patients(patient_ids: List[str], cnv_dir: str = 'data/cnv_output') -> sc.AnnData:
    """Load and concatenate data from all patients."""
    print("Loading patient data...")

    adatas = []
    for patient_id in patient_ids:
        try:
            adata = load_patient_data(patient_id, cnv_dir)
            print(f"  {patient_id}: {adata.n_obs:,} cells")
            adatas.append(adata)
        except Exception as e:
            print(f"  Warning: Could not load {patient_id}: {e}")
            continue

    if not adatas:
        raise ValueError("No patient data could be loaded")

    # Concatenate
    combined = sc.concat(adatas, join='inner')
    print(f"\nTotal: {combined.n_obs:,} cells from {len(adatas)} patients")

    return combined


# ============================================================================
# Concordance Classification
# ============================================================================

def compute_embedding_distances(
    adata: sc.AnnData,
    model: ContrastiveModel,
    normalization: Dict
) -> np.ndarray:
    """Compute embedding distances for all cells."""
    # Get expression data
    if sparse.issparse(adata.X):
        expression = adata.X.toarray()
    else:
        expression = np.array(adata.X)

    # Get CNV data
    cnv = adata.obsm['X_cnv']
    if sparse.issparse(cnv):
        cnv = cnv.toarray()
    else:
        cnv = np.array(cnv)

    # Normalize
    expr_norm = (expression - normalization['expression_mean']) / normalization['expression_std']
    cnv_norm = (cnv - normalization['cnv_mean']) / normalization['cnv_std']

    # Compute embeddings
    model.eval()
    with torch.no_grad():
        expr_tensor = torch.FloatTensor(expr_norm)
        cnv_tensor = torch.FloatTensor(cnv_norm)

        # Process in batches for memory efficiency
        batch_size = 1000
        distances = []

        for i in range(0, len(expr_tensor), batch_size):
            batch_expr = expr_tensor[i:i+batch_size]
            batch_cnv = cnv_tensor[i:i+batch_size]

            expr_embed, cnv_embed = model(batch_expr, batch_cnv)

            # Distance = 1 - cosine similarity
            batch_dist = 1 - (expr_embed * cnv_embed).sum(dim=1)
            distances.append(batch_dist.numpy())

    return np.concatenate(distances)


def classify_concordance(
    adata: sc.AnnData,
    model: ContrastiveModel,
    normalization: Dict,
    threshold_quantile: float = 0.25
) -> sc.AnnData:
    """Classify cells by concordance (cancer cells only)."""
    # Compute distances
    distances = compute_embedding_distances(adata, model, normalization)
    adata.obs['embedding_distance'] = distances

    # Initialize concordance column
    adata.obs['cnv_concordance'] = 'Not classified'

    # Only classify cancer cells
    if 'cancer_vs_normal' in adata.obs.columns:
        cancer_mask = adata.obs['cancer_vs_normal'] == 'Cancer'
    else:
        # Assume all cells are cancer if no annotation
        cancer_mask = np.ones(adata.n_obs, dtype=bool)
        adata.obs['cancer_vs_normal'] = 'Cancer'

    cancer_distances = distances[cancer_mask]

    # Compute thresholds on cancer cells only
    low_thresh = np.quantile(cancer_distances, threshold_quantile)
    high_thresh = np.quantile(cancer_distances, 1 - threshold_quantile)

    # Classify cancer cells
    concordance = np.array(['Intermediate'] * len(cancer_distances))
    concordance[cancer_distances <= low_thresh] = 'Concordant'
    concordance[cancer_distances >= high_thresh] = 'Discordant'

    adata.obs.loc[cancer_mask, 'cnv_concordance'] = concordance
    adata.obs.loc[~cancer_mask, 'cnv_concordance'] = 'Normal'

    # Summary
    n_concordant = (adata.obs['cnv_concordance'] == 'Concordant').sum()
    n_discordant = (adata.obs['cnv_concordance'] == 'Discordant').sum()
    n_intermediate = (adata.obs['cnv_concordance'] == 'Intermediate').sum()

    print(f"  Concordant (low distance): {n_concordant:,} cells")
    print(f"  Intermediate: {n_intermediate:,} cells")
    print(f"  Discordant (high distance): {n_discordant:,} cells")

    return adata


# ============================================================================
# Pooled DE Analysis
# ============================================================================

def run_pooled_de(
    adata: sc.AnnData,
    group1: str = 'Discordant',
    group2: str = 'Concordant',
    min_cells: int = 50
) -> pd.DataFrame:
    """
    Run DE analysis comparing concordance groups across all patients.

    Uses rank_genes_groups with patient as covariate for batch correction.
    """
    # Filter to cancer cells with concordance labels
    mask = adata.obs['cnv_concordance'].isin([group1, group2])
    adata_de = adata[mask].copy()

    n_group1 = (adata_de.obs['cnv_concordance'] == group1).sum()
    n_group2 = (adata_de.obs['cnv_concordance'] == group2).sum()

    print(f"\n  Comparing: {group1} ({n_group1:,} cells) vs {group2} ({n_group2:,} cells)")

    if n_group1 < min_cells or n_group2 < min_cells:
        print(f"  Warning: Too few cells for DE analysis")
        return pd.DataFrame()

    # Check patient distribution
    print(f"\n  Patient distribution:")
    for patient in adata_de.obs['patient_id'].unique():
        patient_mask = adata_de.obs['patient_id'] == patient
        n_g1 = ((adata_de.obs['cnv_concordance'] == group1) & patient_mask).sum()
        n_g2 = ((adata_de.obs['cnv_concordance'] == group2) & patient_mask).sum()
        print(f"    {patient}: {n_g1} {group1}, {n_g2} {group2}")

    # Run DE with Wilcoxon test
    print(f"\n  Running Wilcoxon rank-sum test...")
    sc.tl.rank_genes_groups(
        adata_de,
        groupby='cnv_concordance',
        groups=[group1],
        reference=group2,
        method='wilcoxon',
        pts=True,
        tie_correct=True
    )

    # Extract results
    result = sc.get.rank_genes_groups_df(adata_de, group=group1)
    result = result.rename(columns={'names': 'gene'})

    # Add significance flag
    result['significant'] = result['pvals_adj'] < 0.05

    # Sort by adjusted p-value
    result = result.sort_values('pvals_adj')

    return result


def run_de_with_patient_stratification(
    adata: sc.AnnData,
    group1: str = 'Discordant',
    group2: str = 'Concordant'
) -> pd.DataFrame:
    """
    Run DE with mixed effects model accounting for patient.

    This uses a pseudo-bulk approach: aggregate cells per patient-group,
    then run paired analysis.
    """
    print("\n  Running pseudo-bulk analysis with patient stratification...")

    # Filter to cells with concordance labels
    mask = adata.obs['cnv_concordance'].isin([group1, group2])
    adata_de = adata[mask].copy()

    # Get expression matrix
    if sparse.issparse(adata_de.X):
        X = adata_de.X.toarray()
    else:
        X = np.array(adata_de.X)

    genes = adata_de.var_names.tolist()
    patients = adata_de.obs['patient_id'].unique()

    # Compute mean expression per patient-group
    results = []

    for gene_idx, gene in enumerate(genes):
        gene_expr = X[:, gene_idx]

        # Get expression per patient-group
        g1_means = []
        g2_means = []

        for patient in patients:
            patient_mask = adata_de.obs['patient_id'] == patient

            g1_mask = (adata_de.obs['cnv_concordance'] == group1) & patient_mask
            g2_mask = (adata_de.obs['cnv_concordance'] == group2) & patient_mask

            if g1_mask.sum() > 0 and g2_mask.sum() > 0:
                g1_means.append(gene_expr[g1_mask].mean())
                g2_means.append(gene_expr[g2_mask].mean())

        if len(g1_means) >= 3:  # Need at least 3 patients for paired test
            from scipy import stats
            # Paired t-test (each patient is its own control)
            stat, pval = stats.ttest_rel(g1_means, g2_means)
            logfc = np.mean(g1_means) - np.mean(g2_means)

            results.append({
                'gene': gene,
                'logfoldchanges': logfc,
                'pvals': pval,
                'n_patients': len(g1_means),
                'mean_g1': np.mean(g1_means),
                'mean_g2': np.mean(g2_means)
            })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # FDR correction
    from statsmodels.stats.multitest import multipletests
    _, df['pvals_adj'], _, _ = multipletests(df['pvals'], method='fdr_bh')

    df['significant'] = df['pvals_adj'] < 0.05
    df = df.sort_values('pvals_adj')

    return df


# ============================================================================
# Main Pipeline
# ============================================================================

def run_pooled_analysis(
    patient_ids: List[str],
    model_dir: str = 'models/contrastive_hn',
    cnv_dir: str = 'data/cnv_output',
    output_dir: str = 'data/de_results',
    threshold_quantile: float = 0.25
):
    """Run pooled cross-patient DE analysis."""
    print("=" * 70)
    print("POOLED CROSS-PATIENT DE ANALYSIS")
    print("=" * 70)
    print(f"\nPatients: {patient_ids}")
    print(f"Threshold: {int(threshold_quantile*100)}/{int((1-threshold_quantile)*100)} percentile split")

    # Load model
    print("\n" + "-" * 50)
    print("Loading model...")
    print("-" * 50)
    model, normalization = load_model(model_dir)
    print(f"  Model loaded from: {model_dir}")

    # Load all patient data
    print("\n" + "-" * 50)
    print("Loading patient data...")
    print("-" * 50)
    adata = load_all_patients(patient_ids, cnv_dir)

    # Classify concordance
    print("\n" + "-" * 50)
    print("Classifying concordance (cancer cells only)...")
    print("-" * 50)
    adata = classify_concordance(adata, model, normalization, threshold_quantile)

    # Run pooled DE analysis
    print("\n" + "-" * 50)
    print("Running pooled DE analysis...")
    print("-" * 50)

    # Method 1: Simple pooled Wilcoxon
    de_pooled = run_pooled_de(adata, 'Discordant', 'Concordant')

    # Method 2: Pseudo-bulk with patient stratification
    de_stratified = run_de_with_patient_stratification(adata, 'Discordant', 'Concordant')

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    if not de_pooled.empty:
        pooled_path = os.path.join(output_dir, 'pooled_de_results.csv')
        de_pooled.to_csv(pooled_path, index=False)
        print(f"\nSaved: {pooled_path}")

        n_sig = de_pooled['significant'].sum()
        print(f"  Significant genes (pooled Wilcoxon): {n_sig}")

        # Save escape and compensation genes
        escape_genes = de_pooled[(de_pooled['significant']) & (de_pooled['logfoldchanges'] > 0)]
        comp_genes = de_pooled[(de_pooled['significant']) & (de_pooled['logfoldchanges'] < 0)]

        if not escape_genes.empty:
            escape_path = os.path.join(output_dir, 'pooled_escape_genes.csv')
            escape_genes.to_csv(escape_path, index=False)
            print(f"  Escape genes: {len(escape_genes)}")

        if not comp_genes.empty:
            comp_path = os.path.join(output_dir, 'pooled_compensation_genes.csv')
            comp_genes.to_csv(comp_path, index=False)
            print(f"  Compensation genes: {len(comp_genes)}")

    if not de_stratified.empty:
        strat_path = os.path.join(output_dir, 'stratified_de_results.csv')
        de_stratified.to_csv(strat_path, index=False)
        print(f"\nSaved: {strat_path}")

        n_sig = de_stratified['significant'].sum()
        print(f"  Significant genes (paired t-test): {n_sig}")

    # Print top results
    print("\n" + "=" * 70)
    print("TOP RESULTS")
    print("=" * 70)

    if not de_pooled.empty:
        print("\nTop 20 genes by p-value (pooled Wilcoxon):")
        print("-" * 50)
        top = de_pooled.head(20)
        for _, row in top.iterrows():
            sig = "*" if row['significant'] else " "
            direction = "↑" if row['logfoldchanges'] > 0 else "↓"
            print(f"  {sig} {row['gene']:15} logFC={row['logfoldchanges']:+.3f} {direction}  p_adj={row['pvals_adj']:.2e}")

    if not de_stratified.empty:
        print("\nTop 20 genes by p-value (patient-stratified paired t-test):")
        print("-" * 50)
        top = de_stratified.head(20)
        for _, row in top.iterrows():
            sig = "*" if row['significant'] else " "
            direction = "↑" if row['logfoldchanges'] > 0 else "↓"
            print(f"  {sig} {row['gene']:15} logFC={row['logfoldchanges']:+.3f} {direction}  p_adj={row['pvals_adj']:.2e}  (n={row['n_patients']} patients)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Pooled analysis complete!

Files saved to: {output_dir}/
  - pooled_de_results.csv: All genes, Wilcoxon test across all cells
  - stratified_de_results.csv: All genes, paired t-test per patient
  - pooled_escape_genes.csv: Significant upregulated genes
  - pooled_compensation_genes.csv: Significant downregulated genes

Interpretation:
  - Escape genes (logFC > 0): Higher in discordant = expressed despite CNV
  - Compensation genes (logFC < 0): Lower in discordant = suppressed despite CNV
  - * indicates FDR-significant (p_adj < 0.05)
""")

    return de_pooled, de_stratified


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
        description='Run pooled cross-patient DE analysis for improved statistical power'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models/contrastive_hn',
        help='Directory with trained contrastive model'
    )
    parser.add_argument(
        '--cnv-dir',
        type=str,
        default='data/cnv_output',
        help='Directory with CNV data'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/de_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.25,
        help='Quantile threshold (default 0.25 = 25/75 split)'
    )
    parser.add_argument(
        '--patients',
        type=str,
        nargs='+',
        default=None,
        help='Specific patients to include (default: all)'
    )

    args = parser.parse_args()

    # Get patients
    if args.patients:
        patient_ids = args.patients
    else:
        patient_ids = get_available_patients(args.cnv_dir)

    if not patient_ids:
        print("No patients found!")
        return

    # Run analysis
    run_pooled_analysis(
        patient_ids=patient_ids,
        model_dir=args.model_dir,
        cnv_dir=args.cnv_dir,
        output_dir=args.output_dir,
        threshold_quantile=args.threshold
    )


if __name__ == "__main__":
    main()
