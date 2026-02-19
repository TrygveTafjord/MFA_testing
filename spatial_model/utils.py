import torch

def calculate_rmse(original, reconstructed):
    """Calculates Root Mean Squared Error."""
    mse = torch.mean((original - reconstructed) ** 2, dim=1)
    return torch.sqrt(mse).mean().item()

def calculate_sam(original, reconstructed, epsilon=1e-8):
    """
    Calculates Spectral Angle Mapper (SAM) in radians.
    Formula: arccos( (x . y) / (|x| * |y|) )
    """
    # Normalize vectors to unit length
    norm_orig = torch.norm(original, p=2, dim=1, keepdim=True)
    norm_recon = torch.norm(reconstructed, p=2, dim=1, keepdim=True)
    
    # Dot product
    dot_product = torch.sum(original * reconstructed, dim=1, keepdim=True)
    
    # Cosine similarity
    cosine_sim = dot_product / (norm_orig * norm_recon + epsilon)
    
    # Clamp to avoid numerical issues slightly outside [-1, 1]
    cosine_sim = torch.clamp(cosine_sim, -1.0, 1.0)
    
    sam_rad = torch.acos(cosine_sim)
    return sam_rad.mean().item()