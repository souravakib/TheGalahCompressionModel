#!/usr/bin/env python3
"""
VAE Training without Early Stopping - Focus on Problematic Latent Dimensions
Modified version of galah_variational_autoencoder.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from astropy.io import fits
import warnings
warnings.filterwarnings('ignore')

# Configuration - Focus on problematic latent dimensions
CONFIG = {
    'latent_dims': [128, 256],  # Focus on problematic dimensions that need retraining
    'channels': '32,64,128',  # Match CNN AE channels
    'kernels': '128,64,32',   # Match CNN AE kernels
    'pools_or_strides': '8,8,8',
    'downsampling_mode': 'stride',
    'learning_rate': 1e-5,  # Proven working learning rate
    'batch_size': 32, 
    'num_epochs': 150,  # Match previous successful training
    'beta': 1.0,  # Full KL weight (proven to work)
    'kl_annealing_epochs': 50,  # Match previous successful training
    'free_bits': 0.25,  # Standard free bits
    
    'early_stopping_patience': 0,  # DISABLED - No early stopping
    'base_output_dir': './results/variational_autoencoder_no_early_stop',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'LABEL_CSV': 'C:/E_Drive/kaBHOOM/surrey/dissertation/code/GalahCompressionModel/galah_data/metadata.csv',
    'SPECTRA_DIR': 'C:/E_Drive/kaBHOOM/surrey/dissertation/code/GalahCompressionModel/galah_data/fits',
    'debug_shapes': True,
    'n_spectra': 4000,  # Reduced for faster testing
    'gradient_clip_norm': 1.0,  # Add gradient clipping
    'weight_decay': 1e-5  # Add weight decay
}

class VAEDataset(Dataset):
    """Dataset for VAE training"""
    
    def __init__(self, csv_path, spectra_dir, n_spectra=None):
        self.csv_path = csv_path
        self.spectra_dir = spectra_dir
        
        # Load metadata
        self.df = pd.read_csv(csv_path)
        self.df['sobject_id'] = self.df['sobject_id'].astype(str)
        
        # Limit number of spectra if specified
        if n_spectra is not None:
            self.df = self.df.head(n_spectra)
        
        # Find available spectra
        self.available_files = []
        for idx, row in self.df.iterrows():
            sobject_id = row['sobject_id']
            found = False
            for cam in [1, 2, 3, 4]:
                fpath = os.path.join(spectra_dir, f"{sobject_id}{cam}.fits")
                if os.path.exists(fpath):
                    self.available_files.append(idx)
                    found = True
                    break
            if not found:
                continue
        
        print(f"Found {len(self.available_files)} available spectra")
        
        # Create common wavelength grid
        self.common_wavelength = np.linspace(4000, 8000, 16384)
    
    def __len__(self):
        return len(self.available_files)
    
    def __getitem__(self, idx):
        row_idx = self.available_files[idx]
        row = self.df.iloc[row_idx]
        sobject_id = row['sobject_id']
        
        # Load spectrum
        spectrum = self.get_flux_from_all_cameras(sobject_id)
        
        # Get stellar parameters
        teff = float(row['teff']) if not pd.isna(row['teff']) else 5000.0
        logg = float(row['logg']) if not pd.isna(row['logg']) else 4.0
        fe_h = float(row['Fe_h']) if not pd.isna(row['Fe_h']) else 0.0
        
        # Debug first few samples
        if idx < 3:
            print(f"Sample {idx}: spectrum shape: {spectrum.shape}, range: [{spectrum.min():.3f}, {spectrum.max():.3f}], std: {spectrum.std():.3f}")
        
        return {
            'spectrum': torch.FloatTensor(spectrum).unsqueeze(0),  # Add channel dimension: [1, length]
            'teff': teff,
            'logg': logg,
            'fe_h': fe_h
        }
    
    def get_flux_from_all_cameras(self, sobject_id):
        """Load spectrum from all cameras"""
        try:
            all_wavelengths, all_fluxes = [], []
            
            for cam in [1, 2, 3, 4]:
                fpath = os.path.join(self.spectra_dir, f"{sobject_id}{cam}.fits")
                if os.path.exists(fpath):
                    with fits.open(fpath) as hdul:
                        flux = hdul[4].data
                        hdr = hdul[4].header
                        wave = hdr['CRVAL1'] + hdr['CDELT1'] * np.arange(hdr['NAXIS1'])
                        
                        safe_flux = np.where(flux > 1e-10, flux, 1e-10)
                        flux_processed = np.log(safe_flux)
                        
                        all_wavelengths.append(wave)
                        all_fluxes.append(flux_processed)
            
            if len(all_wavelengths) == 0:
                return np.zeros(16384)
            
            # Concatenate and sort
            full_wave = np.concatenate(all_wavelengths)
            full_flux = np.concatenate(all_fluxes)
            sort_idx = np.argsort(full_wave)
            full_wave = full_wave[sort_idx]
            full_flux = full_flux[sort_idx]
            
            # Interpolate to common grid
            interp_func = interp1d(full_wave, full_flux, kind='linear', 
                                  bounds_error=False, fill_value=0.0)
            interp_flux = interp_func(self.common_wavelength)
            
            # Normalize
            if np.std(interp_flux) > 0:
                interp_flux = (interp_flux - np.mean(interp_flux)) / np.std(interp_flux)
            
            return interp_flux
            
        except Exception as e:
            return np.zeros(16384)

class VAE(nn.Module):
    """Variational Autoencoder"""
    
    def __init__(self, input_length, latent_dim, channels, kernels, pools_or_strides):
        super().__init__()
        self.input_length = input_length
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        in_channels = 1
        
        for i in range(len(channels)):
            conv_layer = nn.Conv1d(in_channels, channels[i], kernel_size=kernels[i],
                                 stride=pools_or_strides[i], padding=(kernels[i] - 1) // 2)
            encoder_layers.extend([conv_layer, nn.BatchNorm1d(channels[i]), nn.ReLU()])
            in_channels = channels[i]
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Calculate flattened size and store encoder output shape
        with torch.no_grad():
            dummy_output = self.encoder(torch.zeros(1, 1, input_length))
        self.flattened_size = dummy_output.flatten(start_dim=1).shape[1]
        self.encoder_output_channels = dummy_output.shape[1]  # Store number of channels from encoder
        
        # Latent space
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)
        
        # Decoder
        self.fc_decoder = nn.Linear(latent_dim, self.flattened_size)
        
        decoder_layers = []
        in_channels = channels[-1]
        
        for i in range(len(channels) - 1, -1, -1):
            if i == 0:
                out_channels = 1
            else:
                out_channels = channels[i-1]
            
            # Calculate output padding for proper size
            stride = pools_or_strides[i]
            output_padding = (input_length - (input_length // stride) * stride) % stride
            output_padding = min(output_padding, stride - 1)  # Cap at stride-1
            
            conv_transpose = nn.ConvTranspose1d(in_channels, out_channels, 
                                              kernel_size=kernels[i], 
                                              stride=stride, 
                                              padding=(kernels[i] - 1) // 2,
                                              output_padding=output_padding)
            decoder_layers.append(conv_transpose)
            
            if i > 0:
                decoder_layers.extend([nn.BatchNorm1d(out_channels), nn.ReLU()])
            
            in_channels = out_channels
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.fc_decoder(z)
        h = h.view(h.size(0), -1, 1)
        h = h.view(h.size(0), self.encoder_output_channels, -1)  # Use stored encoder output channels
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def vae_loss(recon_x, x, mu, logvar, beta=1.0, free_bits=0.25):
    """VAE loss function with free bits"""
    # Reconstruction loss
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence loss with free bits
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = torch.clamp(kl_div - free_bits, min=0.0)
    
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

def train_vae_for_latent_dim(latent_dim, config, dataset=None):
    """Train VAE for specific latent dimension"""
    
    print(f"\n{'='*60}")
    print(f"Training VAE with Latent Dimension {latent_dim}")
    print(f"{'='*60}")
    
    # Create output directory
    output_dir = os.path.join(config['base_output_dir'], f'L_{latent_dim}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse architecture parameters
    channels = [int(x) for x in config['channels'].split(',')]
    kernels = [int(x) for x in config['kernels'].split(',')]
    pools_or_strides = [int(x) for x in config['pools_or_strides'].split(',')]
    
    # Create dataset if not provided
    if dataset is None:
        dataset = VAEDataset(config['LABEL_CSV'], config['SPECTRA_DIR'], config['n_spectra'])
    
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    
    # Create model
    model = VAE(16384, latent_dim, channels, kernels, pools_or_strides)
    model = model.to(config['device'])
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer with standard learning rate
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    # Training history
    train_losses = []
    val_losses = []
    recon_losses = []
    kl_losses = []
    beta_values = []
    
    # Training loop
    model.train()
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['num_epochs']):
        # KL annealing with standard settings
        if epoch < config['kl_annealing_epochs']:
            beta = config['beta'] * (epoch / config['kl_annealing_epochs'])
        else:
            beta = config['beta']
        
        beta_values.append(beta)
        
        # Training
        epoch_train_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            spectrum = batch['spectrum'].to(config['device'])
            
            optimizer.zero_grad()
            
            # Forward pass
            recon_spectrum, mu, logvar = model(spectrum)
            
            # Debug shapes and values
            if batch_idx == 0 and epoch == 0:
                print(f"Debug - Input shape: {spectrum.shape}, range: [{spectrum.min():.3f}, {spectrum.max():.3f}]")
                print(f"Debug - Output shape: {recon_spectrum.shape}, range: [{recon_spectrum.min():.3f}, {recon_spectrum.max():.3f}]")
                print(f"Debug - Mu shape: {mu.shape}, range: [{mu.min():.3f}, {mu.max():.3f}]")
                print(f"Debug - Logvar shape: {logvar.shape}, range: [{logvar.min():.3f}, {logvar.max():.3f}]")
            
            # Handle size mismatch
            if recon_spectrum.shape[-1] != spectrum.shape[-1]:
                min_len = min(recon_spectrum.shape[-1], spectrum.shape[-1])
                recon_spectrum = recon_spectrum[..., :min_len]
                spectrum = spectrum[..., :min_len]
                if batch_idx == 0 and epoch == 0:
                    print(f"Debug - Size mismatch fixed: {recon_spectrum.shape} vs {spectrum.shape}")
            
            # Loss
            loss, recon_loss, kl_loss = vae_loss(recon_spectrum, spectrum, mu, logvar, beta, config['free_bits'])
            
            # Debug loss values
            if batch_idx == 0 and epoch == 0:
                print(f"Debug - Total loss: {loss.item():.6f}")
                print(f"Debug - Recon loss: {recon_loss.item():.6f}")
                print(f"Debug - KL loss: {kl_loss.item():.6f}")
                print(f"Debug - Beta: {beta:.6f}")
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'])
            
            optimizer.step()
            
            epoch_train_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1}/{config["num_epochs"]}, Batch {batch_idx}/{len(dataloader)}, '
                      f'Loss: {loss.item():.6f}, Recon: {recon_loss.item():.6f}, KL: {kl_loss.item():.6f}')
        
        # Average losses
        avg_train_loss = epoch_train_loss / len(dataloader)
        avg_recon_loss = epoch_recon_loss / len(dataloader)
        avg_kl_loss = epoch_kl_loss / len(dataloader)
        
        train_losses.append(avg_train_loss)
        recon_losses.append(avg_recon_loss)
        kl_losses.append(avg_kl_loss)
        
        # Validation (use training data for simplicity)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                spectrum = batch['spectrum'].to(config['device'])
                recon_spectrum, mu, logvar = model(spectrum)
                
                if recon_spectrum.shape[-1] != spectrum.shape[-1]:
                    min_len = min(recon_spectrum.shape[-1], spectrum.shape[-1])
                    recon_spectrum = recon_spectrum[..., :min_len]
                    spectrum = spectrum[..., :min_len]
                
                loss, _, _ = vae_loss(recon_spectrum, spectrum, mu, logvar, beta, config['free_bits'])
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(dataloader)
        val_losses.append(avg_val_loss)
        
        model.train()
        
        print(f'Epoch {epoch+1}/{config["num_epochs"]}: Train Loss: {avg_train_loss:.6f}, '
              f'Val Loss: {avg_val_loss:.6f}, Beta: {beta:.4f}')
        
        # Save best model (but no early stopping)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': avg_val_loss,
            }, os.path.join(output_dir, 'best_model.pth'))
        
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': avg_val_loss,
            }, os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': config['num_epochs'],
        'loss': avg_val_loss,
    }, os.path.join(output_dir, 'final_model.pth'))
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'recon_losses': recon_losses,
        'kl_losses': kl_losses,
        'beta_values': beta_values,
        'best_val_loss': best_val_loss
    }
    
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training history
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].plot(train_losses, label='Train Loss')
    axes[0, 0].plot(val_losses, label='Val Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(recon_losses, label='Reconstruction Loss')
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(kl_losses, label='KL Loss')
    axes[1, 0].set_title('KL Divergence Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(beta_values, label='Beta')
    axes[1, 1].set_title('KL Weight (Beta)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nTraining completed for L{latent_dim}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Results saved to: {output_dir}")
    
    return model, history

def main():
    """Main training function"""
    
    print("VAE Training without Early Stopping")
    print("=" * 60)
    print(f"Latent dimensions: {CONFIG['latent_dims']}")
    print(f"Epochs: {CONFIG['num_epochs']}")
    print(f"Learning rate: {CONFIG['learning_rate']}")
    print(f"Early stopping: DISABLED")
    print("=" * 60)
    
    # Create base output directory
    os.makedirs(CONFIG['base_output_dir'], exist_ok=True)
    
    # Train each latent dimension
    for latent_dim in CONFIG['latent_dims']:
        try:
            model, history = train_vae_for_latent_dim(latent_dim, CONFIG)
            print(f"✅ L{latent_dim}: Training completed successfully")
        except Exception as e:
            print(f"❌ L{latent_dim}: Training failed - {str(e)}")
    
    print("\n" + "=" * 60)
    print("ALL TRAINING COMPLETED!")
    print("=" * 60)

if __name__ == "__main__":
    main()
