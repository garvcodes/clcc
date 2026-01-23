"""
Data processing utilities and PyTorch dataset class.
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import scanpy as sc
from typing import Tuple, Optional


class MultimodalScDataset(Dataset):
    """
    PyTorch dataset for paired expression-CNV data.
    
    Each sample returns:
        - x_expr: normalized expression vector (n_genes,)
        - x_cnv: CNV profile for the cell's subcluster (n_genes,)
        - label: subcluster index
    """
    
    def __init__(
        self,
        adata,
        cnv_profiles,
        subcluster_col='subcluster',
        expr_layer=None
    ):
        """
        Args:
            adata: AnnData object with expression data
            cnv_profiles: DataFrame or array (n_subclusters, n_genes) with CNV profiles
            subcluster_col: column name in adata.obs containing subcluster labels
            expr_layer: which layer to use (None = .X, 'normalized' = .layers['normalized'])
        """
        self.adata = adata
        self.cnv_profiles = cnv_profiles
        self.subcluster_col = subcluster_col
        self.expr_layer = expr_layer
        
        # Get subcluster labels
        self.labels = adata.obs[subcluster_col].values
        
        # Create mapping from subcluster name to index
        unique_subclusters = sorted(adata.obs[subcluster_col].unique())
        self.subcluster_to_idx = {sc: idx for idx, sc in enumerate(unique_subclusters)}
        self.label_indices = np.array([self.subcluster_to_idx[label] for label in self.labels])
        
        # Store CNV profiles as tensor
        if isinstance(cnv_profiles, pd.DataFrame):
            cnv_profiles = cnv_profiles.values
        self.cnv_tensor = torch.FloatTensor(cnv_profiles)
        
        print(f"Dataset initialized:")
        print(f"  {len(self)} cells")
        print(f"  {len(unique_subclusters)} subclusters")
        print(f"  {self.cnv_tensor.shape[1]} genes")
        
    def __len__(self):
        return len(self.adata)
    
    def __getitem__(self, idx):
        # Get expression vector
        if self.expr_layer is None:
            x_expr = self.adata.X[idx].toarray().flatten() if hasattr(self.adata.X[idx], 'toarray') else self.adata.X[idx]
        else:
            x_expr = self.adata.layers[self.expr_layer][idx]
            if hasattr(x_expr, 'toarray'):
                x_expr = x_expr.toarray().flatten()
        
        # Get CNV profile for this cell's subcluster
        subcluster_idx = self.label_indices[idx]
        x_cnv = self.cnv_tensor[subcluster_idx]
        
        return {
            'x_expr': torch.FloatTensor(x_expr),
            'x_cnv': x_cnv,
            'label': subcluster_idx
        }


def preprocess_adata(
    adata,
    min_genes=200,
    min_cells=3,
    target_sum=1e4,
    n_top_genes=None,
    log_transform=True
):
    """
    Standard preprocessing pipeline for scRNA-seq data.
    
    Args:
        adata: AnnData object
        min_genes: minimum genes per cell
        min_cells: minimum cells per gene
        target_sum: target sum for normalization
        n_top_genes: keep only top variable genes (None = keep all)
        log_transform: whether to log-transform
        
    Returns:
        adata: preprocessed AnnData object
    """
    print("Starting preprocessing...")
    print(f"Initial shape: {adata.shape}")
    
    # Filter cells and genes
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    print(f"After filtering: {adata.shape}")
    
    # Calculate QC metrics
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)
    
    # Normalize
    sc.pp.normalize_total(adata, target_sum=target_sum)
    
    if log_transform:
        sc.pp.log1p(adata)
    
    # Select highly variable genes
    if n_top_genes is not None:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
        adata = adata[:, adata.var.highly_variable].copy()
        print(f"After HVG selection: {adata.shape}")
    
    print("Preprocessing complete!")
    return adata


def create_gene_position_file(gene_list, output_path, assembly='hg38'):
    """
    Create gene position file for inferCNV using MyGene.info.
    
    Args:
        gene_list: list of gene symbols
        output_path: where to save the file
        assembly: genome assembly (hg38 or hg19)
        
    Returns:
        DataFrame with columns: gene, chr, start, end
    """
    import mygene
    
    print(f"Querying gene positions for {len(gene_list)} genes...")
    mg = mygene.MyGeneInfo()
    
    # Query in batches
    results = mg.querymany(
        gene_list,
        scopes='symbol',
        fields='genomic_pos',
        species='human',
        assembly=assembly
    )
    
    # Parse results
    gene_positions = []
    for result in results:
        if 'genomic_pos' in result:
            pos = result['genomic_pos']
            if isinstance(pos, list):
                pos = pos[0]  # Take first position if multiple
            
            # Skip non-standard chromosomes
            chr_name = str(pos.get('chr', ''))
            if chr_name.startswith('chr'):
                chr_name = chr_name[3:]
            
            if chr_name in [str(i) for i in range(1, 23)] + ['X', 'Y']:
                gene_positions.append({
                    'gene': result['query'],
                    'chr': chr_name,
                    'start': pos.get('start', 0),
                    'end': pos.get('end', 0)
                })
    
    # Create DataFrame
    df = pd.DataFrame(gene_positions)
    
    # Sort by chromosome and position
    df['chr_num'] = df['chr'].apply(lambda x: int(x) if x.isdigit() else (23 if x == 'X' else 24))
    df = df.sort_values(['chr_num', 'start']).drop('chr_num', axis=1)
    
    # Save
    df.to_csv(output_path, sep='\t', index=False, header=False)
    print(f"Saved gene positions for {len(df)} genes to {output_path}")
    
    return df


if __name__ == "__main__":
    # Test dataset creation with dummy data
    print("Creating dummy data for testing...")
    
    # Create dummy AnnData
    n_cells = 1000
    n_genes = 100
    n_subclusters = 10
    
    X = np.random.randn(n_cells, n_genes)
    obs = pd.DataFrame({
        'subcluster': np.random.choice([f'cluster_{i}' for i in range(n_subclusters)], n_cells)
    })
    var = pd.DataFrame(index=[f'gene_{i}' for i in range(n_genes)])
    
    adata = sc.AnnData(X=X, obs=obs, var=var)
    
    # Create dummy CNV profiles
    cnv_profiles = np.random.randn(n_subclusters, n_genes)
    
    # Create dataset
    dataset = MultimodalScDataset(adata, cnv_profiles)
    
    # Test getting an item
    sample = dataset[0]
    print(f"\nSample shapes:")
    print(f"  x_expr: {sample['x_expr'].shape}")
    print(f"  x_cnv: {sample['x_cnv'].shape}")
    print(f"  label: {sample['label']}")
