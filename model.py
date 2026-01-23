"""
Model architectures for multimodal contrastive learning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """3-layer MLP encoder with ReLU and batch normalization."""
    
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        
        # 3-layer MLP
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        # No BN or ReLU on final layer
        
    def forward(self, x):
        # Layer 1
        x = self.layer1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Layer 2
        x = self.layer2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Layer 3 (no activation)
        x = self.layer3(x)
        
        return x  # h-space representation (B, 256)


class ProjectionHead(nn.Module):
    """Projection head: 256 -> 128 -> 64 with L2 normalization."""
    
    def __init__(self, hidden_dim=256, intermediate_dim=128, output_dim=64):
        super().__init__()
        
        self.fc1 = nn.Linear(hidden_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, output_dim)
        
    def forward(self, h):
        # 256 -> 128 with ReLU
        z = self.fc1(h)
        z = F.relu(z)
        
        # 128 -> 64
        z = self.fc2(z)
        
        # L2 normalize
        z = F.normalize(z, p=2, dim=1)
        
        return z  # z-space representation (B, 64)


class MultimodalEncoder(nn.Module):
    """Complete multimodal encoder with expression and CNV pathways."""
    
    def __init__(self, n_genes, hidden_dim=256, latent_dim=64, freeze_cnv=True):
        super().__init__()
        
        # Expression encoder (trainable)
        self.expr_encoder = Encoder(n_genes, hidden_dim)
        self.expr_projection = ProjectionHead(hidden_dim, 128, latent_dim)
        
        # CNV encoder (frozen)
        self.cnv_encoder = Encoder(n_genes, hidden_dim)
        self.cnv_projection = ProjectionHead(hidden_dim, 128, latent_dim)
        
        if freeze_cnv:
            # Freeze CNV pathway
            for param in self.cnv_encoder.parameters():
                param.requires_grad = False
            for param in self.cnv_projection.parameters():
                param.requires_grad = False
                
    def forward_expression(self, x_expr):
        """Forward pass for expression data."""
        h_expr = self.expr_encoder(x_expr)
        z_expr = self.expr_projection(h_expr)
        return h_expr, z_expr
    
    def forward_cnv(self, x_cnv):
        """Forward pass for CNV data (no gradients)."""
        with torch.no_grad():
            h_cnv = self.cnv_encoder(x_cnv)
            z_cnv = self.cnv_projection(h_cnv)
        return h_cnv, z_cnv
    
    def forward(self, x_expr, x_cnv):
        """Forward pass for both modalities."""
        h_expr, z_expr = self.forward_expression(x_expr)
        h_cnv, z_cnv = self.forward_cnv(x_cnv)
        
        return {
            'h_expr': h_expr,  # (B, 256)
            'z_expr': z_expr,  # (B, 64)
            'h_cnv': h_cnv,    # (B, 256)
            'z_cnv': z_cnv     # (B, 64)
        }


def build_cnv_anchor_bank(model, cnv_profiles, device):
    """
    Build bank of CNV anchors for all subclusters.
    
    Args:
        model: MultimodalEncoder
        cnv_profiles: tensor of shape (M, n_genes) for M subclusters
        device: torch device
        
    Returns:
        Z_cnv: tensor of shape (M, latent_dim) - normalized CNV embeddings
    """
    model.eval()
    with torch.no_grad():
        cnv_profiles = cnv_profiles.to(device)
        _, z_cnv = model.forward_cnv(cnv_profiles)
    return z_cnv


if __name__ == "__main__":
    # Test the model
    n_genes = 5884
    batch_size = 32
    n_subclusters = 78
    
    # Create dummy data
    x_expr = torch.randn(batch_size, n_genes)
    x_cnv = torch.randn(batch_size, n_genes)
    
    # Initialize model
    model = MultimodalEncoder(n_genes)
    
    # Forward pass
    outputs = model(x_expr, x_cnv)
    
    print("Output shapes:")
    for key, val in outputs.items():
        print(f"  {key}: {val.shape}")
    
    # Test anchor bank
    cnv_profiles = torch.randn(n_subclusters, n_genes)
    anchor_bank = build_cnv_anchor_bank(model, cnv_profiles, 'cpu')
    print(f"\nAnchor bank shape: {anchor_bank.shape}")
