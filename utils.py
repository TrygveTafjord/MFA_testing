import torch
import torch.nn.functional as F
from hypso import Hypso
import numpy as np


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


def extract_spatial_features(hsi_cube_np, window_size=3):

    # Reshape to (Batch, Channels, Height, Width) for PyTorch 2D operations
    cube_t = torch.from_numpy(hsi_cube_np).float().unsqueeze(0).permute(0, 3, 1, 2)
    
    pad = window_size // 2
    
    avg_pool = F.avg_pool2d(cube_t, kernel_size=window_size, stride=1, padding=pad)
    
    augmented_cube = torch.cat([cube_t, avg_pool], dim=1)
    
    return augmented_cube.squeeze(0).permute(1, 2, 0).numpy()



def get_data(data_dir, data_product, target_total_samples, ADD_SPATIAL_INFO):
    
    len_data_dir = len(data_dir)
    samples_per_file = target_total_samples // len_data_dir  
    sampled_data_list = []

    print(f"Aiming to extract ~{samples_per_file} pixels per file from {len_data_dir} files to reach a total of ~{target_total_samples} samples.")

    i = 0
    for file in data_dir:
        # Load Data
        try:
            satobj = Hypso(file) 
            if satobj is None: continue

            # Load and reshape
            match data_product:
                case 'l1a':
                    data = satobj.l1a_cube.values.astype(np.float32)
                case 'l1b':
                    data = satobj.l1b_cube.values.astype(np.float32)
                case 'l1d':
                    data = satobj.l1d_cube.values.astype(np.float32)
                case _:
                    raise ValueError(f"Unknown data product: {data_product}")

            if ADD_SPATIAL_INFO:
                data = extract_spatial_features(data)

            h, w, b = data.shape
            data_2d = data.reshape(-1, b) # Shape: (Total_Pixels_In_Image, 120)

            # Random Subsampling 
            total_pixels_in_image = data_2d.shape[0]

            # Determine how many to take (don't take more than exists)
            n_to_take = min(samples_per_file, total_pixels_in_image)

            # Generate random indices
            rng = np.random.default_rng()
            indices = rng.choice(total_pixels_in_image, size=n_to_take, replace=False)

            # Grab the random pixels and add to list
            sampled_pixel_subset = data_2d[indices, :]
            sampled_data_list.append(sampled_pixel_subset)

            print(f"{i}/{len_data_dir} | File: {file} | Extracted {n_to_take} pixels.")
            i += 1
        except Exception as e:
            print(f"Error processing {file}: {e}")
        
    data = np.concatenate(sampled_data_list, axis=0)
    #Convert to torch tensor
    print("-" * 30)
    print(f"Final Analysis Dataset Shape: {data.shape}")

    return data 
