"""
Evaluation metrics and visualization for multimodal contrastive learning.
"""
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import scanpy as sc


def compute_embeddings(model, dataloader, device='cuda'):
    """
    Compute embeddings for all cells.
    
    Args:
        model: MultimodalEncoder
        dataloader: DataLoader
        device: torch device
        
    Returns:
        h_expr: (N, 256) h-space embeddings
        z_expr: (N, 64) z-space embeddings
        labels: (N,) subcluster labels
    """
    model.eval()
    
    h_expr_list = []
    z_expr_list = []
    labels_list = []
    
    with torch.no_grad():
        for batch in dataloader:
            x_expr = batch['x_expr'].to(device)
            labels = batch['label'].cpu().numpy()
            
            # Forward pass (only expression)
            h_expr, z_expr = model.forward_expression(x_expr)
            
            h_expr_list.append(h_expr.cpu().numpy())
            z_expr_list.append(z_expr.cpu().numpy())
            labels_list.append(labels)
    
    h_expr = np.vstack(h_expr_list)
    z_expr = np.vstack(z_expr_list)
    labels = np.concatenate(labels_list)
    
    return h_expr, z_expr, labels


def compute_subcluster_centroids(embeddings, labels):
    """
    Compute centroid for each subcluster.
    
    Args:
        embeddings: (N, d) embeddings
        labels: (N,) subcluster labels
        
    Returns:
        centroids: (M, d) centroids
        unique_labels: (M,) unique labels in order
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    d = embeddings.shape[1]
    
    centroids = np.zeros((n_clusters, d))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        centroids[i] = embeddings[mask].mean(axis=0)
    
    return centroids, unique_labels


def top_k_retrieval_accuracy(expr_centroids, cnv_anchors, k=5):
    """
    Compute top-k retrieval accuracy.
    
    For each expression centroid, check if the correct CNV anchor
    is among the top-k most similar anchors.
    
    Args:
        expr_centroids: (M, d) expression centroids
        cnv_anchors: (M, d) CNV anchors (in same order)
        k: top-k threshold
        
    Returns:
        accuracy: fraction of centroids that retrieve correct anchor in top-k
        similarities: (M, M) cosine similarity matrix
    """
    # Normalize
    expr_centroids_norm = expr_centroids / np.linalg.norm(expr_centroids, axis=1, keepdims=True)
    cnv_anchors_norm = cnv_anchors / np.linalg.norm(cnv_anchors, axis=1, keepdims=True)
    
    # Compute cosine similarity
    similarities = expr_centroids_norm @ cnv_anchors_norm.T  # (M, M)
    
    # For each row, check if diagonal element is in top-k
    n_clusters = len(expr_centroids)
    correct = 0
    
    for i in range(n_clusters):
        # Get top-k indices for this row
        top_k_indices = np.argsort(similarities[i])[-k:]
        
        # Check if correct index (i) is in top-k
        if i in top_k_indices:
            correct += 1
    
    accuracy = correct / n_clusters
    return accuracy, similarities


def evaluate_alignment(model, dataset, cnv_profiles_tensor, device='cuda', k=5):
    """
    Evaluate expression-CNV alignment using top-k retrieval.
    
    Args:
        model: MultimodalEncoder
        dataset: MultimodalScDataset
        cnv_profiles_tensor: (M, n_genes) CNV profiles
        device: torch device
        k: top-k threshold
        
    Returns:
        results: dict with evaluation metrics
    """
    from torch.utils.data import DataLoader
    from model import build_cnv_anchor_bank
    
    print("Computing embeddings...")
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False)
    h_expr, z_expr, labels = compute_embeddings(model, dataloader, device)
    
    print("Computing CNV anchors...")
    model.eval()
    with torch.no_grad():
        h_cnv = model.cnv_encoder(cnv_profiles_tensor.to(device)).cpu().numpy()
        z_cnv = model.cnv_projection(torch.FloatTensor(h_cnv).to(device)).cpu().numpy()
    
    print("Computing centroids...")
    z_expr_centroids, unique_labels = compute_subcluster_centroids(z_expr, labels)
    h_expr_centroids, _ = compute_subcluster_centroids(h_expr, labels)
    
    print(f"Evaluating top-{k} retrieval...")
    z_accuracy, z_similarities = top_k_retrieval_accuracy(z_expr_centroids, z_cnv, k=k)
    h_accuracy, h_similarities = top_k_retrieval_accuracy(h_expr_centroids, h_cnv, k=k)
    
    results = {
        'z_space_accuracy': z_accuracy,
        'h_space_accuracy': h_accuracy,
        'z_similarities': z_similarities,
        'h_similarities': h_similarities,
        'z_expr': z_expr,
        'h_expr': h_expr,
        'z_expr_centroids': z_expr_centroids,
        'h_expr_centroids': h_expr_centroids,
        'z_cnv': z_cnv,
        'h_cnv': h_cnv,
        'labels': labels
    }
    
    print(f"\nResults:")
    print(f"  Z-space top-{k} accuracy: {z_accuracy:.1%}")
    print(f"  H-space top-{k} accuracy: {h_accuracy:.1%}")
    
    return results


def plot_similarity_heatmap(similarities, save_path=None, title="CNV Similarity Matrix"):
    """
    Plot cosine similarity heatmap.
    
    Args:
        similarities: (M, M) similarity matrix
        save_path: where to save figure
        title: plot title
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        similarities,
        cmap='viridis',
        center=0.98,
        vmin=similarities.min(),
        vmax=1.0,
        square=True,
        cbar_kws={'label': 'Cosine Similarity'}
    )
    plt.title(title)
    plt.xlabel("CNV Anchor Index")
    plt.ylabel("Expression Centroid Index")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved similarity heatmap to {save_path}")
    
    plt.show()


def plot_umap(
    embeddings,
    labels,
    color_by='cluster',
    title="UMAP of Embeddings",
    save_path=None,
    n_neighbors=30,
    min_dist=0.3
):
    """
    Plot UMAP visualization of embeddings.
    
    Args:
        embeddings: (N, d) embeddings
        labels: (N,) labels for coloring
        color_by: what to color by
        title: plot title
        save_path: where to save
        n_neighbors: UMAP parameter
        min_dist: UMAP parameter
    """
    import umap
    
    print("Computing UMAP...")
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=labels,
        cmap='tab20',
        s=1,
        alpha=0.6
    )
    plt.colorbar(scatter, label=color_by)
    plt.title(title)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved UMAP to {save_path}")
    
    plt.show()
    
    return embedding_2d


def plot_training_curves(history, save_path=None):
    """
    Plot training loss curves.
    
    Args:
        history: dict with loss history
        save_path: where to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Total loss
    axes[0, 0].plot(history['total_loss'])
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    
    # Contrastive loss
    axes[0, 1].plot(history['contrastive_loss'])
    axes[0, 1].set_title('Contrastive Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    
    # Centroid loss
    axes[1, 0].plot(history['centroid_loss'])
    axes[1, 0].set_title('Centroid Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    
    # H-space alignment loss
    axes[1, 1].plot(history['h_align_loss'])
    axes[1, 1].set_title('H-space Alignment Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    print("This module contains evaluation functions.")
    print("Import and use them in your analysis notebooks.")
