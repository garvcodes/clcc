"""
Loss functions for multimodal contrastive learning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def info_nce_loss(z_expr, z_cnv_anchors, labels, temperature=0.2, top_k=10):
    """
    InfoNCE contrastive loss with hard negative mining.
    
    Args:
        z_expr: (B, d) normalized expression embeddings
        z_cnv_anchors: (M, d) normalized CNV anchor embeddings
        labels: (B,) subcluster labels for each expression sample
        temperature: temperature parameter
        top_k: number of hard negatives to keep per sample
        
    Returns:
        loss: scalar loss value
    """
    # Compute cosine similarity logits: (B, M)
    logits = torch.matmul(z_expr, z_cnv_anchors.T) / temperature
    
    batch_size = z_expr.size(0)
    n_anchors = z_cnv_anchors.size(0)
    
    # Create mask for positive pairs
    pos_mask = torch.zeros(batch_size, n_anchors, device=z_expr.device)
    pos_mask[torch.arange(batch_size), labels] = 1
    
    # Extract positive logits
    pos_logits = logits[pos_mask.bool()].view(batch_size, 1)
    
    # Hard negative mining: select top-k hardest negatives per sample
    neg_mask = 1 - pos_mask
    neg_logits = logits.clone()
    neg_logits[pos_mask.bool()] = float('-inf')  # Mask out positives
    
    # Get top-k hardest negatives (highest logits)
    top_k = min(top_k, n_anchors - 1)  # Ensure we don't select more than available
    hard_neg_logits, _ = torch.topk(neg_logits, k=top_k, dim=1)
    
    # Combine positive and hard negatives: (B, 1+k)
    combined_logits = torch.cat([pos_logits, hard_neg_logits], dim=1)
    
    # Target is always index 0 (positive)
    targets = torch.zeros(batch_size, dtype=torch.long, device=z_expr.device)
    
    # Cross-entropy loss
    loss = F.cross_entropy(combined_logits, targets)
    
    return loss


def centroid_regularization_loss(embeddings, labels, centroids=None):
    """
    Centroid regularization: encourage samples to stay close to their subcluster centroid.
    
    Args:
        embeddings: (B, d) embeddings (either h-space or z-space)
        labels: (B,) subcluster labels
        centroids: (M, d) precomputed centroids (optional)
        
    Returns:
        loss: scalar loss value
        centroids: (M, d) computed centroids
    """
    if centroids is None:
        # Compute centroids on-the-fly
        unique_labels = torch.unique(labels)
        n_clusters = len(unique_labels)
        d = embeddings.size(1)
        
        centroids = torch.zeros(n_clusters, d, device=embeddings.device)
        for i, label in enumerate(unique_labels):
            mask = labels == label
            centroids[i] = embeddings[mask].mean(dim=0)
    
    # Get centroid for each sample
    sample_centroids = centroids[labels]
    
    # MSE loss between embeddings and their centroids
    loss = F.mse_loss(embeddings, sample_centroids)
    
    return loss, centroids


def h_space_alignment_loss(h_expr, h_cnv):
    """
    H-space alignment: align intermediate representations.
    
    Args:
        h_expr: (B, 256) expression h-space embeddings
        h_cnv: (B, 256) CNV h-space embeddings
        
    Returns:
        loss: scalar MSE loss
    """
    return F.mse_loss(h_expr, h_cnv)


class CombinedLoss(nn.Module):
    """Combined loss with all three components."""
    
    def __init__(
        self,
        temperature=0.2,
        top_k_negatives=10,
        lambda_centroid=0.05,
        lambda_h_align=0.1,
        use_z_centroid=True
    ):
        super().__init__()
        self.temperature = temperature
        self.top_k_negatives = top_k_negatives
        self.lambda_centroid = lambda_centroid
        self.lambda_h_align = lambda_h_align
        self.use_z_centroid = use_z_centroid
        
        # Cache for centroids (updated each epoch)
        self.z_centroids = None
        self.h_centroids = None
        
    def forward(self, outputs, z_cnv_anchors, labels):
        """
        Compute combined loss.
        
        Args:
            outputs: dict with keys 'h_expr', 'z_expr', 'h_cnv', 'z_cnv'
            z_cnv_anchors: (M, latent_dim) bank of CNV anchors
            labels: (B,) subcluster labels
            
        Returns:
            total_loss: scalar
            loss_dict: dict with individual loss components
        """
        z_expr = outputs['z_expr']
        h_expr = outputs['h_expr']
        h_cnv = outputs['h_cnv']
        
        # 1. Contrastive loss
        loss_contrastive = info_nce_loss(
            z_expr, 
            z_cnv_anchors, 
            labels, 
            temperature=self.temperature,
            top_k=self.top_k_negatives
        )
        
        # 2. Centroid regularization
        if self.use_z_centroid:
            loss_centroid, self.z_centroids = centroid_regularization_loss(
                z_expr, labels, self.z_centroids
            )
        else:
            loss_centroid, self.h_centroids = centroid_regularization_loss(
                h_expr, labels, self.h_centroids
            )
        
        # 3. H-space alignment
        loss_h_align = h_space_alignment_loss(h_expr, h_cnv)
        
        # Combined loss
        total_loss = (
            loss_contrastive + 
            self.lambda_centroid * loss_centroid +
            self.lambda_h_align * loss_h_align
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'contrastive': loss_contrastive.item(),
            'centroid': loss_centroid.item(),
            'h_align': loss_h_align.item()
        }
        
        return total_loss, loss_dict
    
    def reset_centroids(self):
        """Reset cached centroids (call at start of each epoch)."""
        self.z_centroids = None
        self.h_centroids = None


if __name__ == "__main__":
    # Test losses
    batch_size = 32
    latent_dim = 64
    hidden_dim = 256
    n_subclusters = 78
    
    # Create dummy data
    z_expr = F.normalize(torch.randn(batch_size, latent_dim), p=2, dim=1)
    h_expr = torch.randn(batch_size, hidden_dim)
    h_cnv = torch.randn(batch_size, hidden_dim)
    z_cnv_anchors = F.normalize(torch.randn(n_subclusters, latent_dim), p=2, dim=1)
    labels = torch.randint(0, n_subclusters, (batch_size,))
    
    outputs = {
        'z_expr': z_expr,
        'h_expr': h_expr,
        'h_cnv': h_cnv,
        'z_cnv': None  # Not used
    }
    
    # Test combined loss
    loss_fn = CombinedLoss()
    total_loss, loss_dict = loss_fn(outputs, z_cnv_anchors, labels)
    
    print("Loss components:")
    for key, val in loss_dict.items():
        print(f"  {key}: {val:.4f}")
