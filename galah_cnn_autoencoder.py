import os
import random
import json
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime

# --- Configuration ---
config = {
    "experiment_name": "cnn_ae_pym_l_32",
    "downsampling_mode": "stride",
    "pools_or_strides": "8,8,8", 
    "kernels": "128,64,32",
    "channels": "32,64,128",
    "latent_dim": 32,
    "batch_size": 32,
    "epochs": 50,
    "lr": 1e-4,
    "early_stopping_patience": 10,
    "early_stopping_min_delta": 0.0001,
    "n_spectra": 4000,
    "output_dir": "./results/unsupervised_cnn_ae/L_32",
    "random_seed": 42
}

# List of latent dimensions to experiment with
LATENT_DIMS = [16, 32, 64, 128, 256]

LABEL_CSV = 'C:/E_Drive/kaBHOOM/surrey/dissertation/code/GalahCompressionModel/galah_data/metadata.csv'
SPECTRA_DIR = 'C:/E_Drive/kaBHOOM/surrey/dissertation/code/GalahCompressionModel/galah_data/fits'

# Seeding
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Data loading from fits files ---
def get_flux_from_all_cameras(sobject_id, spectra_dir):
    all_wavelengths, all_fluxes = [], []
    for cam in [1, 2, 3, 4]:
        fpath = os.path.join(spectra_dir, f"{sobject_id}{cam}.fits")
        try:
            if not os.path.exists(fpath):
                return None, None
            with fits.open(fpath) as hdul:
                flux = hdul[4].data
                hdr = hdul[4].header
                wave = hdr['CRVAL1'] + hdr['CDELT1'] * np.arange(hdr['NAXIS1'])
                safe_flux = np.where(flux > 1e-10, flux, 1e-10)
                flux_processed = np.log(safe_flux)
                all_wavelengths.append(wave)
                all_fluxes.append(flux_processed)
        except Exception as e:
            return None, None
    full_wave = np.concatenate(all_wavelengths)
    full_flux = np.concatenate(all_fluxes)
    sort_idx = np.argsort(full_wave)
    return full_wave[sort_idx], full_flux[sort_idx]

# Dataset class
class SpectraDataset(Dataset):
    def __init__(self, sobject_ids, spectra_dir, common_grid):
        self.sobject_ids = sobject_ids
        self.spectra_dir = spectra_dir
        self.common_grid = torch.tensor(common_grid, dtype=torch.float32)
    def __len__(self):
        return len(self.sobject_ids)
    def __getitem__(self, idx):
        sobject_id = self.sobject_ids[idx]
        wave, flux = get_flux_from_all_cameras(sobject_id, self.spectra_dir)
        if flux is None:
            return torch.zeros_like(self.common_grid), torch.zeros_like(self.common_grid)
        interp_func = interp1d(wave, flux, kind='linear', bounds_error=False, fill_value=0.0)
        interp_flux = interp_func(self.common_grid.numpy())
        mask = (self.common_grid.numpy() >= wave.min()) & (self.common_grid.numpy() <= wave.max())
        return torch.tensor(interp_flux, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

# CNN Autoencoder
class CNNAutoencoder(nn.Module):
    def __init__(self, config, debug_shapes=False):
        super().__init__()
        channels = [int(c) for c in config['channels'].split(',')]
        kernels = [int(k) for k in config['kernels'].split(',')]
        strides = [int(s) for s in config['pools_or_strides'].split(',')]
        num_layers = len(channels)
        input_length = config['input_length']
        
        self.debug_shapes = debug_shapes
        self.input_length = input_length  # Store for adaptive pooling
        
        # Encoder
        self.encoder_layers = nn.ModuleList()
        in_channels = 1
        current_length = input_length
        encoder_output_lengths = []
        
        if debug_shapes:
            print(f"Encoder size tracking:")
            print(f"  Input length: {input_length}")
        
        for i, (c, k, s) in enumerate(zip(channels, kernels, strides)):
            padding =  (k - 1) // 2
            self.encoder_layers.append(
                nn.Conv1d(in_channels, c, kernel_size=k, stride=s, padding=padding)
            )
            self.encoder_layers.append(nn.BatchNorm1d(c))
            self.encoder_layers.append(nn.ReLU())
            
            # Calculate output size: (input_size + 2*padding - kernel_size) / stride + 1
            current_length = (current_length + (2 * padding) - k) // s + 1
            encoder_output_lengths.append(current_length)
            
            if debug_shapes:
                print(f"  Layer {i}: {current_length} (kernel={k}, stride={s}, padding={padding})")
            
            in_channels = c
        
        self.enc_out_channels = channels[-1]
        self.enc_out_length = encoder_output_lengths[-1]
        flattened_size = self.enc_out_channels * self.enc_out_length
        
        self.to_latent = nn.Linear(flattened_size, config['latent_dim'])
        self.from_latent = nn.Linear(config['latent_dim'], flattened_size)
        
        # Decoder
        rev_channels = list(reversed(channels))
        rev_kernels = list(reversed(kernels))
        rev_strides = list(reversed(strides))
        rev_encoder_output_lengths = list(reversed(encoder_output_lengths))
        
        self.decoder_layers = nn.ModuleList()
        
        if debug_shapes:
            print(f"Decoder size tracking:")
            print(f"  Target output length: {input_length}")
        
        for i in range(num_layers):
            in_ch = rev_channels[i]
            if i == num_layers - 1:
                out_ch = 1
                target_length = input_length
            else:
                out_ch = rev_channels[i + 1]
                target_length = rev_encoder_output_lengths[i + 1]
            
            k = rev_kernels[i]
            s = rev_strides[i]
            padding =  (k - 1) // 2
            
            input_len = rev_encoder_output_lengths[i]
            
            # Calculate output_padding for ConvTranspose1d
            output_padding = target_length - (((input_len - 1) * s) + k - (2 * padding))
            
            if debug_shapes:
                print(f"  Decoder layer {i}: input_len={input_len}, target={target_length}, output_padding={output_padding}")
            
            # Ensure output_padding is valid (0 or 1 for ConvTranspose1d)
            if output_padding < 0:
                # If negative, we need to adjust the target length or use a different approach
                output_padding = 0
                print(f"Warning: Adjusted output_padding to 0 for decoder layer {i}")
            elif output_padding > 1:
                # If too large, cap it at 1
                output_padding = 1
                print(f"Warning: Capped output_padding to 1 for decoder layer {i}")
            
            assert output_padding in [0, 1], f"Output padding {output_padding} invalid at decoder layer {i}"
            
            self.decoder_layers.append(
                nn.ConvTranspose1d(
                    in_ch,
                    out_ch,
                    kernel_size=k,
                    stride=s,
                    padding=padding,
                    output_padding=output_padding
                )
            )
            
            if i != num_layers - 1:
                self.decoder_layers.append(nn.BatchNorm1d(out_ch))
                self.decoder_layers.append(nn.ReLU())
    
    def forward(self, x):
        # Only print debug shapes during initialization, not during training
        if self.debug_shapes and hasattr(self, '_debug_printed'):
            return self._forward_without_debug(x)
        
        if self.debug_shapes:
            print(f"Input shape: {x.shape}")
        
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)
            if self.debug_shapes:
                print(f"Encoder Layer {i} ({layer.__class__.__name__}): {x.shape}")
        
        x = x.flatten(1)
        if self.debug_shapes:
            print(f"Flattened shape: {x.shape}")
        
        latent = self.to_latent(x)
        if self.debug_shapes:
            print(f"Latent vector shape: {latent.shape}")
        
        x = self.from_latent(latent)
        x = x.view(-1, self.enc_out_channels, self.enc_out_length)
        if self.debug_shapes:
            print(f"From latent reshaped: {x.shape}")
        
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x)
            if self.debug_shapes:
                print(f"Decoder Layer {i} ({layer.__class__.__name__}): {x.shape}")
        
        # Mark that debug has been printed
        if self.debug_shapes:
            self._debug_printed = True
        
        return x.squeeze(1), latent
    
    def _forward_without_debug(self, x):
        """Forward pass without debug printing for training"""
        for layer in self.encoder_layers:
            x = layer(x)
        
        x = x.flatten(1)
        latent = self.to_latent(x)
        x = self.from_latent(latent)
        x = x.view(-1, self.enc_out_channels, self.enc_out_length)
        
        for layer in self.decoder_layers:
            x = layer(x)
        
        return x.squeeze(1), latent

# --- Main training function ---
def main():
    set_seed(config['random_seed'])
    
    # Train multiple models with different latent dimensions
    all_results = {}
    
    for latent_dim in LATENT_DIMS:
        try:
            results = train_single_model(latent_dim, config)
            if results:
                all_results[latent_dim] = results
        except Exception as e:
            print(f"Error training model with latent dimension {latent_dim}: {e}")
            continue
    
    # Save summary of all experiments
    if all_results:
        summary_path = "./results/unsupervised_cnn_ae/experiment_summary.json"
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        
        summary = {
            'experiment_date': str(datetime.now()),
            'models_trained': len(all_results),
            'latent_dimensions': list(all_results.keys()),
            'results_summary': {
                str(latent_dim): {
                    'best_val_loss': results['best_val_loss'],
                    'final_epoch': results['final_epoch'],
                    'early_stopped': results['early_stopped']
                }
                for latent_dim, results in all_results.items()
            }
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        # Create comparison plot
        plt.figure(figsize=(12, 8))
        for latent_dim, results in all_results.items():
            plt.plot(results['val_loss_history'], label=f'L={latent_dim}', marker='o', markersize=3)
        
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.title('Comparison of Validation Loss Across Latent Dimensions')
        plt.legend()
        plt.grid(True)
        plt.savefig("./results/unsupervised_cnn_ae/comparison_plot.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n{'='*60}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        print(f"Models trained: {len(all_results)}")
        print(f"Latent dimensions: {list(all_results.keys())}")
        print("\nBest validation losses:")
        for latent_dim, results in all_results.items():
            print(f"  L={latent_dim}: {results['best_val_loss']:.6f}")
        print(f"\nSummary saved to: {summary_path}")
        print(f"Comparison plot saved to: ./results/unsupervised_cnn_ae/comparison_plot.png")
    else:
        print("No models were successfully trained.")

def train_single_model(latent_dim, base_config):
    """Train a single model with given latent dimension and save results"""
    
    # Create experiment-specific config
    experiment_config = base_config.copy()
    experiment_config['latent_dim'] = latent_dim
    experiment_config['experiment_name'] = f"cnn_ae_pym_l_{latent_dim}"
    experiment_config['output_dir'] = f"./results/unsupervised_cnn_ae/L_{latent_dim}"
    
    print(f"\n{'='*60}")
    print(f"Training model with latent dimension: {latent_dim}")
    print(f"Output directory: {experiment_config['output_dir']}")
    print(f"{'='*60}")
    
    # Create output directory
    os.makedirs(experiment_config['output_dir'], exist_ok=True)
    
    # Check if spectra directory exists
    if not os.path.exists(SPECTRA_DIR):
        print(f"Error: Spectra directory not found: {SPECTRA_DIR}")
        return None
    
    # Load and prepare data
    df = pd.read_csv(LABEL_CSV)
    df['sobject_id'] = df['sobject_id'].astype(str)
    if experiment_config['n_spectra'] is not None:
        df = df.head(experiment_config['n_spectra'])
    
    # Prepare common grid
    sample_ids = df['sobject_id'].sample(n=min(50, len(df)), random_state=experiment_config['random_seed']).tolist()
    sample_waves = [wave for sid in sample_ids if (wave := get_flux_from_all_cameras(sid, SPECTRA_DIR)[0]) is not None]
    common_grid = sample_waves[np.argsort([len(w) for w in sample_waves])[len(sample_waves)//2]]
    experiment_config['input_length'] = len(common_grid)
    print(f"Using common grid length: {experiment_config['input_length']}")
    
    # Split data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=experiment_config['random_seed'])
    train_dataset = SpectraDataset(train_df['sobject_id'].tolist(), SPECTRA_DIR, common_grid)
    val_dataset = SpectraDataset(val_df['sobject_id'].tolist(), SPECTRA_DIR, common_grid)
    
    train_loader = DataLoader(train_dataset, batch_size=experiment_config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=experiment_config['batch_size'], shuffle=False)
    
    # Setup model and training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNAutoencoder(experiment_config, debug_shapes=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=experiment_config['lr'])
    
    # Training variables
    best_val_loss = float('inf')
    patience_counter = 0
    loss_history, val_loss_history = [], []
    
    # Training loop
    for epoch in range(experiment_config['epochs']):
        model.train()
        running_loss = 0.0
        size_mismatch_reported = False
        
        for batch, mask in train_loader:
            batch, mask = batch.to(device), mask.to(device)
            optimizer.zero_grad()
            output, _ = model(batch.unsqueeze(1))
            
            # Handle size mismatch
            min_size = min(output.shape[1], batch.shape[1])
            if output.shape[1] != batch.shape[1] and not size_mismatch_reported:
                print(f"Epoch {epoch+1}: Size mismatch: output={output.shape[1]}, batch={batch.shape[1]}, using first {min_size} elements")
                size_mismatch_reported = True
            
            # Loss calculation
            output_trimmed = output[:, :min_size]
            batch_trimmed = batch[:, :min_size]
            mask_trimmed = mask[:, :min_size]
            
            loss = (((output_trimmed - batch_trimmed) ** 2) * mask_trimmed).sum() / mask_trimmed.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        loss_history.append(train_loss)
        
        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch, mask in val_loader:
                batch, mask = batch.to(device), mask.to(device)
                output, _ = model(batch.unsqueeze(1))
                
                min_size = min(output.shape[1], batch.shape[1])
                output_trimmed = output[:, :min_size]
                batch_trimmed = batch[:, :min_size]
                mask_trimmed = mask[:, :min_size]
                
                val_loss = (((output_trimmed - batch_trimmed) ** 2) * mask_trimmed).sum() / mask_trimmed.sum()
                running_val_loss += val_loss.item()
        
        val_loss = running_val_loss / len(val_loader)
        val_loss_history.append(val_loss)
        
        print(f"Epoch {epoch+1}/{experiment_config['epochs']} — Train Loss: {train_loss:.6f} — Val Loss: {val_loss:.6f}")
        
        # Early stopping
        if best_val_loss - val_loss > experiment_config['early_stopping_min_delta']:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(experiment_config['output_dir'], 'best_model.pth'))
            print("Validation loss improved; model saved.")
        else:
            patience_counter += 1
            print(f"No improvement; patience {patience_counter}/{experiment_config['early_stopping_patience']}")
            if patience_counter >= experiment_config['early_stopping_patience']:
                print("Early stopping triggered.")
                break
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(experiment_config['output_dir'], 'final_model.pth'))
    
    # Save training logs
    results = {
        'config': experiment_config,
        'train_loss_history': loss_history,
        'val_loss_history': val_loss_history,
        'best_val_loss': best_val_loss,
        'final_epoch': len(loss_history),
        'early_stopped': patience_counter >= experiment_config['early_stopping_patience']
    }
    
    with open(os.path.join(experiment_config['output_dir'], 'training_logs.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Save loss curve plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_history) + 1), loss_history, label='Train Loss', marker='o')
    plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label='Val Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Masked MSE Loss')
    plt.title(f'Training & Validation Loss (Latent Dim: {latent_dim})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(experiment_config['output_dir'], 'loss_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training completed for latent dimension {latent_dim}")
    print(f"Results saved to: {experiment_config['output_dir']}")
    
    return results

if __name__ == '__main__':
    main()
