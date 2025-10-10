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

# Configuration
CONFIG = {
    'latent_dims': [64, 512, 1024],  # Match CNN AE latent dimensions
    'channels': '32,64,128',  # Match CNN AE channels
    'kernels': '128,64,32',   # Match CNN AE kernels
    'pools_or_strides': '8,8,8',
    'downsampling_mode': 'stride',
    'learning_rate': 1e-5,  # More conservative learning rate for stability
    'batch_size': 32,
    'num_epochs': 150,
    'beta': 1.0,  # Final KL divergence weight (can be tuned)
    'kl_annealing_epochs': 50,  # Number of epochs to gradually increase KL weight
    'free_bits': 0.25,  # KL divergence free bits (allows small KL without penalty)
    'early_stopping_patience': 0,  # DISABLED - no early stopping
    'base_output_dir': './results/variational_autoencoder_v7',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'LABEL_CSV': '/user/HS401/ss06019/Desktop/dissertation/GalahCompressionModel/galah_all.csv',
    'SPECTRA_DIR': '/user/HS401/ss06019/Desktop/dissertation/GalahCompressionModel/galah_data/fits',
    'debug_shapes': True,  # Set to False for laptop training to reduce console output
    'n_spectra': 4000  # Match CNN AE number of spectra
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
        
        print(f"Dataset size: {len(self.df)}")
        
        # Filter for available spectra
        self.available_files = []
        for idx, row in self.df.iterrows():
            sobject_id = row['sobject_id']
            # Check if any of the FITS files exist (with digits 1-4 appended)
            found = False
            for digit in range(1, 5):  # Try digits 1-4
                fpath = os.path.join(spectra_dir, f"{sobject_id}{digit}.fits")
                if os.path.exists(fpath):
                    self.available_files.append(idx)
                    found = True
                    break
            if not found:
                # Also try without digit (original pattern)
                fpath = os.path.join(spectra_dir, f"{sobject_id}.fits")
                if os.path.exists(fpath):
                    self.available_files.append(idx)
        
        print(f"Available spectra: {len(self.available_files)}")
        
        # Sample some spectra to get actual wavelength ranges
        sample_ids = self.df['sobject_id'].sample(n=min(50, len(self.df)), random_state=42).tolist()
        
        # Sample actual wavelengths first
        sample_waves = []
        for sid in sample_ids:
            result = self.get_flux_from_all_cameras(sid, return_raw=True)
            if result[0] is not None:
                sample_waves.append(result[0])
        
        if sample_waves:
            # Use median length spectrum as common grid (same logic as CNN AE)
            common_grid = sample_waves[np.argsort([len(w) for w in sample_waves])[len(sample_waves)//2]]
            self.common_wavelength = common_grid
            self.input_length = len(common_grid)
            print(f"Using sampled wavelength grid: {len(common_grid)} points")
        else:
            # If sampling fails, create an extended grid that covers all cameras
            # This ensures we don't miss Camera 4 (Near-IR: 7585-7886 Å)
            extended_wavelength = np.linspace(4000, 8000, 20000)
            self.common_wavelength = extended_wavelength
            self.input_length = 20000
            print("Sampling failed, using extended wavelength range (4000-8000 Å) to cover all cameras")
        
        print(f"Using common grid length: {self.input_length}")
        print(f"Wavelength range: {self.common_wavelength.min():.2f} - {self.common_wavelength.max():.2f} Å")
        
        # Debug: Show what wavelength ranges were found in samples
        if sample_waves:
            print(f"Sample wavelength ranges found:")
            for i, wave in enumerate(sample_waves[:5]):  # Show first 5 samples
                print(f"  Sample {i+1}: {wave.min():.2f} - {wave.max():.2f} Å (length: {len(wave)})")
            
            # Check camera coverage
            print(f"\nCamera coverage analysis:")
            camera_ranges = {
                "Camera 1 (Blue)": (4000, 5800),
                "Camera 2 (Green)": (5700, 6800), 
                "Camera 3 (Red)": (6800, 7900),
                "Camera 4 (Near-IR)": (7585, 7886)
            }
            
            for camera_name, (cam_min, cam_max) in camera_ranges.items():
                covered = (self.common_wavelength.min() <= cam_min) and (self.common_wavelength.max() >= cam_max)
                print(f"  {camera_name}: {cam_min}-{cam_max} Å - {'COVERED' if covered else 'NOT COVERED'}")
        else:
            print("No sample wavelengths found, using extended range (4000-8000 Å)")
        
    def __len__(self):
        return len(self.available_files)
    
    def get_flux_from_all_cameras(self, sobject_id, return_raw=False):
        """Extract flux from all cameras and interpolate to common grid"""
        try:
            all_wavelengths, all_fluxes = [], []
            
            # Try to find FITS files with digits 1-4 appended (like the regular autoencoder)
            for cam in [1, 2, 3, 4]:
                fpath = os.path.join(self.spectra_dir, f"{sobject_id}{cam}.fits")
                try:
                    if not os.path.exists(fpath):
                        continue
                    with fits.open(fpath) as hdul:
                        flux = hdul[4].data
                        hdr = hdul[4].header
                        wave = hdr['CRVAL1'] + hdr['CDELT1'] * np.arange(hdr['NAXIS1'])
                        safe_flux = np.where(flux > 1e-10, flux, 1e-10)
                        flux_processed = np.log(safe_flux)
                        all_wavelengths.append(wave)
                        all_fluxes.append(flux_processed)
                except Exception as e:
                    continue
            
            if len(all_wavelengths) == 0:
                return None, None, None
            
            # Concatenate all data
            full_wave = np.concatenate(all_wavelengths)
            full_flux = np.concatenate(all_fluxes)
            sort_idx = np.argsort(full_wave)
            wavelengths = full_wave[sort_idx]
            fluxes = full_flux[sort_idx]
            
            # During sampling phase, just return raw wavelengths and fluxes
            if return_raw:
                return wavelengths, fluxes, None
            
            # During normal operation, interpolate to common grid
            if len(wavelengths) > 10 and hasattr(self, 'common_wavelength'):
                interp_func = interp1d(wavelengths, fluxes, kind='linear', 
                                     bounds_error=False, fill_value=0.0)
                interpolated_flux = interp_func(self.common_wavelength)
                
                # Create mask for valid wavelengths (like CNN AE)
                mask = (self.common_wavelength >= wavelengths.min()) & (self.common_wavelength <= wavelengths.max())
                
                return self.common_wavelength, interpolated_flux, mask
            else:
                return None, None, None
                
        except Exception as e:
            return None, None, None
    
    def __getitem__(self, idx):
        file_idx = self.available_files[idx]
        row = self.df.iloc[file_idx]
        sobject_id = row['sobject_id']
        
        # Get spectrum and mask
        result = self.get_flux_from_all_cameras(sobject_id)
        
        if result[0] is None:
            # Return zero spectrum and mask if file not found
            flux = np.zeros(self.input_length)
            mask = np.zeros(self.input_length)
        else:
            wavelength, flux, mask = result
        
        # Convert to tensor
        flux_tensor = torch.FloatTensor(flux).unsqueeze(0)  # Add channel dimension
        mask_tensor = torch.FloatTensor(mask)
        
        return flux_tensor, mask_tensor

class VAE(nn.Module):
    """Variational Autoencoder for stellar spectra"""
    
    def __init__(self, config, input_length, debug_shapes=False):
        super().__init__()
        
        # Parse configuration
        channels = [int(c) for c in config['channels'].split(',')]
        kernels = [int(k) for k in config['kernels'].split(',')]
        pools_or_strides = [int(p) for p in config['pools_or_strides'].split(',')]
        downsampling_mode = config['downsampling_mode']
        latent_dim = config['latent_dim']
        
        self.latent_dim = latent_dim
        self.input_length = input_length
        self.debug_shapes = debug_shapes
        
        # Encoder
        encoder_layers = []
        in_channels = 1
        current_length = input_length
        encoder_output_lengths = []
        
        if debug_shapes:
            print(f"Encoder size tracking:")
            print(f"  Input length: {input_length}")
        
        for i in range(len(channels)):
            k = kernels[i]
            s = pools_or_strides[i] if downsampling_mode == 'stride' else 1
            padding = (k - 1) // 2
            
            conv_layer = nn.Conv1d(in_channels, channels[i], kernel_size=k,
                                   stride=s, padding=padding)
            encoder_layers.extend([conv_layer, nn.BatchNorm1d(channels[i]), nn.ReLU()])
            
            # Calculate output size: (input_size + 2*padding - kernel_size) / stride + 1
            current_length = (current_length + (2 * padding) - k) // s + 1
            encoder_output_lengths.append(current_length)
            
            if debug_shapes:
                print(f"  Layer {i}: {current_length} (kernel={k}, stride={s}, padding={padding})")
            
            in_channels = channels[i]
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.enc_out_channels = channels[-1]
        self.enc_out_length = encoder_output_lengths[-1]
        self.flattened_size = self.enc_out_channels * self.enc_out_length
        
        if debug_shapes:
            print(f"Encoder output shape: {self.enc_out_channels} x {self.enc_out_length}")
            print(f"Flattened size: {self.flattened_size}")
        
        # Latent space (mu and logvar)
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)
        
        # Decoder
        decoder_layers = []
        # Fix the channel reversal - channels should go from encoder output back to 1
        rev_channels = list(reversed(channels)) + [1]  # Add 1 at the end for final output
        rev_in_channels = list(reversed(channels))
        rev_kernels = list(reversed(kernels))
        rev_strides = list(reversed(pools_or_strides))
        rev_encoder_output_lengths = list(reversed(encoder_output_lengths))
        
        # From latent to decoder input
        self.fc_decoder = nn.Linear(latent_dim, self.flattened_size)
        
        if debug_shapes:
            print(f"Decoder size tracking:")
            print(f"  Target output length: {input_length}")
        
        for i in range(len(channels)):
            in_ch = rev_in_channels[i]
            if i == len(channels) - 1:
                out_ch = 1
                target_length = input_length
            else:
                out_ch = rev_channels[i + 1]
                target_length = rev_encoder_output_lengths[i + 1]
            
            k = rev_kernels[i]
            s = rev_strides[i]
            padding = (k - 1) // 2
            
            input_len = rev_encoder_output_lengths[i]
            
            # Calculate output_padding for ConvTranspose1d
            output_padding = target_length - (((input_len - 1) * s) + k - (2 * padding))
            
            if debug_shapes:
                print(f"  Decoder layer {i}: input_len={input_len}, target={target_length}, output_padding={output_padding}")
            
            # Ensure output_padding is valid (0 or 1 for ConvTranspose1d)
            if output_padding < 0:
                output_padding = 0
                print(f"Warning: Adjusted output_padding to 0 for decoder layer {i}")
            elif output_padding > 1:
                output_padding = 1
                print(f"Warning: Capped output_padding to 1 for decoder layer {i}")
            
            assert output_padding in [0, 1], f"Output padding {output_padding} invalid at decoder layer {i}"
            
            decoder_layers.append(
                nn.ConvTranspose1d(
                    in_ch,
                    out_ch,
                    kernel_size=k,
                    stride=s,
                    padding=padding,
                    output_padding=output_padding
                )
            )
            
            if i != len(channels) - 1:
                decoder_layers.extend([nn.BatchNorm1d(out_ch), nn.ReLU()])
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Store reshape info
        self.reshape_shape = (1, self.enc_out_channels, self.enc_out_length)
        
        # Save architecture info
        self.architecture_info = {
            'input_length': input_length,
            'latent_dim': latent_dim,
            'channels': channels,
            'kernels': kernels,
            'pools_or_strides': pools_or_strides,
            'downsampling_mode': downsampling_mode,
            'flattened_size': self.flattened_size,
            'reshape_shape': list(self.reshape_shape),
            'total_parameters': sum(p.numel() for p in self.parameters())
        }
    
    def encode(self, x):
        """Encode input to latent parameters"""
        if self.debug_shapes:
            print(f"Input shape: {x.shape}")
        
        x = self.encoder(x)
        if self.debug_shapes:
            print(f"Encoder output shape: {x.shape}")
        
        x = x.flatten(start_dim=1)
        if self.debug_shapes:
            print(f"Flattened shape: {x.shape}")
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        if self.debug_shapes:
            print(f"Mu shape: {mu.shape}")
            print(f"Logvar shape: {logvar.shape}")
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent vector to output"""
        if self.debug_shapes:
            print(f"Latent input shape: {z.shape}")
        
        x = self.fc_decoder(z)
        if self.debug_shapes:
            print(f"Decoder input shape: {x.shape}")
        
        x = x.view(x.size(0), *self.reshape_shape[1:])
        if self.debug_shapes:
            print(f"Reshaped shape: {x.shape}")
        
        x = self.decoder(x)
        if self.debug_shapes:
            print(f"Decoder output shape: {x.shape}")
        
        return x
    
    def forward(self, x):
        """Forward pass"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

class VAETrainer:
    """Trainer for VAE"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config['device']
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        
        # Loss function
        self.reconstruction_loss = nn.MSELoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.recon_losses = []
        self.kl_losses = []
        self.beta_values = []  # Track beta values for KL annealing
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def loss_function(self, recon_x, x, mask, mu, logvar, beta=1.0, free_bits=0.25):
        """VAE loss: reconstruction + KL divergence with masking + free bits"""
        # Handle size mismatch between reconstruction and input
        min_size = min(recon_x.size(-1), x.size(-1))
        recon_x_trimmed = recon_x[..., :min_size]
        x_trimmed = x[..., :min_size]
        mask_trimmed = mask[..., :min_size]
        
        # Masked reconstruction loss (like CNN AE)
        # Handle case where mask might be all zeros
        if mask_trimmed.sum() > 0:
            recon_loss = (((recon_x_trimmed - x_trimmed) ** 2) * mask_trimmed).sum() / mask_trimmed.sum()
        else:
            # Fallback to unmasked loss if mask is all zeros
            recon_loss = ((recon_x_trimmed - x_trimmed) ** 2).mean()
        
        # KL divergence with free bits
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Apply free bits: only penalize KL divergence above the threshold
        # Ensure free_bits is a tensor for proper broadcasting
        if isinstance(free_bits, (int, float)):
            free_bits = torch.tensor(free_bits, device=kl_loss.device, dtype=kl_loss.dtype)
        kl_loss = torch.clamp(kl_loss, min=free_bits)
        
        # Total loss
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def train_epoch(self, train_loader):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        for batch_idx, (data, mask) in enumerate(train_loader):
            data, mask = data.to(self.device), mask.to(self.device)
            self.optimizer.zero_grad()
            
            # Forward pass
            recon_batch, mu, logvar = self.model(data)
            
            # Calculate loss with free bits and current beta (for KL annealing)
            current_beta = self.config.get('current_beta', self.config['beta'])
            loss, recon_loss, kl_loss = self.loss_function(recon_batch, data, mask, mu, logvar, 
                                                         current_beta, self.config['free_bits'])
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            
            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")
        
        return (total_loss / len(train_loader), 
                total_recon_loss / len(train_loader), 
                total_kl_loss / len(train_loader))
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        with torch.no_grad():
            for data, mask in val_loader:
                data, mask = data.to(self.device), mask.to(self.device)
                recon_batch, mu, logvar = self.model(data)
                # Calculate loss with free bits and current beta (for KL annealing)
                current_beta = self.config.get('current_beta', self.config['beta'])
                loss, recon_loss, kl_loss = self.loss_function(recon_batch, data, mask, mu, logvar, 
                                                             current_beta, self.config['free_bits'])
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
        
        return (total_loss / len(val_loader), 
                total_recon_loss / len(val_loader), 
                total_kl_loss / len(val_loader))
    
    def train(self, train_loader, val_loader, output_dir):
        """Train the model with early stopping and KL annealing"""
        print(f"=== Training VAE with Latent Dim {self.model.latent_dim} ===")
        print(f"KL Annealing: {self.config['kl_annealing_epochs']} epochs")
        print(f"Free Bits: {self.config['free_bits']}")
        print(f"Early Stopping: {'DISABLED' if self.config['early_stopping_patience'] == 0 else f'Patience={self.config['early_stopping_patience']}'}")
        
        for epoch in range(self.config['num_epochs']):
            # Calculate KL annealing weight
            if epoch < self.config['kl_annealing_epochs']:
                # Gradually increase beta from 0 to final value
                current_beta = (epoch / self.config['kl_annealing_epochs']) * self.config['beta']
            else:
                current_beta = self.config['beta']
            
            # Update config for this epoch
            self.config['current_beta'] = current_beta
            
            # Train
            train_loss, train_recon, train_kl = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_recon, val_kl = self.validate(val_loader)
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.recon_losses.append(train_recon)
            self.kl_losses.append(train_kl)
            self.beta_values.append(current_beta)
            
            # Print progress
            if (epoch + 1) % 1 == 0:  # Print every epoch
                print(f'Epoch [{epoch+1}/{self.config["num_epochs"]}]')
                print(f'Beta: {current_beta:.4f} (KL weight)')
                print(f'Train Loss: {train_loss:.6f} (Recon: {train_recon:.6f}, KL: {train_kl:.6f})')
                print(f'Val Loss: {val_loss:.6f} (Recon: {val_recon:.6f}, KL: {val_kl:.6f})')
                print('-' * 50)
            
            # Early stopping - ONLY if patience > 0
            if self.config['early_stopping_patience'] > 0:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), 
                              os.path.join(output_dir, 'best_model.pth'))
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.config['early_stopping_patience']:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                # No early stopping - always save the latest model
                torch.save(self.model.state_dict(), 
                          os.path.join(output_dir, 'best_model.pth'))
        
        print("Training completed!")
    
    def plot_training_history(self, output_dir):
        """Plot training history with KL annealing"""
        plt.figure(figsize=(20, 5))
        
        # Total loss
        plt.subplot(1, 4, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Reconstruction loss
        plt.subplot(1, 4, 2)
        plt.plot(self.recon_losses, label='Reconstruction Loss')
        plt.title('Reconstruction Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # KL divergence
        plt.subplot(1, 4, 3)
        plt.plot(self.kl_losses, label='KL Divergence', color='red')
        plt.title('KL Divergence')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Beta values (KL annealing)
        plt.subplot(1, 4, 4)
        plt.plot(self.beta_values, label='Beta (KL Weight)', color='green', linewidth=2)
        plt.title('KL Annealing Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Beta Value')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_samples(self, num_samples=10):
        """Generate samples from the VAE"""
        self.model.eval()
        with torch.no_grad():
            # Sample from prior
            z = torch.randn(num_samples, self.model.latent_dim).to(self.device)
            # Decode
            samples = self.model.decode(z)
        return samples.cpu().numpy()
    
    def plot_reconstructions(self, test_loader, output_dir, num_samples=5):
        """Plot original vs reconstructed spectra"""
        self.model.eval()
        
        # Get some test samples
        test_data, test_mask = next(iter(test_loader))
        test_data = test_data[:num_samples].to(self.device)
        test_mask = test_mask[:num_samples].to(self.device)
        
        with torch.no_grad():
            recon_data, mu, logvar = self.model(test_data)
        
        # Plot
        plt.figure(figsize=(15, 10))
        
        for i in range(num_samples):
            plt.subplot(num_samples, 2, 2*i + 1)
            plt.plot(test_data[i, 0].cpu().numpy(), label='Original', alpha=0.7)
            plt.title(f'Sample {i+1} - Original')
            plt.legend()
            
            plt.subplot(num_samples, 2, 2*i + 2)
            plt.plot(recon_data[i, 0].cpu().numpy(), label='Reconstructed', alpha=0.7)
            plt.title(f'Sample {i+1} - Reconstructed')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'reconstructions.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_latent_space(self, test_loader, output_dir, num_samples=1000):
        """Plot 2D projection of latent space"""
        self.model.eval()
        
        # Collect latent representations
        mus = []
        logvars = []
        
        with torch.no_grad():
            for i, (data, mask) in enumerate(test_loader):
                if i * self.config['batch_size'] >= num_samples:
                    break
                data = data.to(self.device)
                mu, logvar = self.model.encode(data)
                mus.append(mu.cpu())
                logvars.append(logvar.cpu())
        
        mus = torch.cat(mus, dim=0)[:num_samples]
        logvars = torch.cat(logvars, dim=0)[:num_samples]
        
        # Plot mu values
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(mus[:, 0], mus[:, 1], alpha=0.6)
        plt.title(f'Latent Space (mu) - First 2 Dimensions (L_{self.model.latent_dim})')
        plt.xlabel('mu_1')
        plt.ylabel('mu_2')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.scatter(logvars[:, 0], logvars[:, 1], alpha=0.6)
        plt.title(f'Latent Space (log variance) - First 2 Dimensions (L_{self.model.latent_dim})')
        plt.xlabel('log variance_1')
        plt.ylabel('log variance_2')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'latent_space.png'), dpi=300, bbox_inches='tight')
        plt.close()

def train_vae_for_latent_dim(latent_dim, config, dataset):
    """Train VAE for a specific latent dimension"""
    
    # Create output directory for this latent dimension
    output_dir = os.path.join(config['base_output_dir'], f'L_{latent_dim}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Update config for this specific model
    model_config = config.copy()
    model_config['latent_dim'] = latent_dim
    model_config['output_dir'] = output_dir
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    print(f"\n=== Training VAE with Latent Dimension {latent_dim} ===")
    print(f"Dynamic input length: {dataset.input_length}")
    print(f"Wavelength range: {dataset.common_wavelength.min():.2f} - {dataset.common_wavelength.max():.2f} Å")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create model with dynamic input length
    model = VAE(model_config, dataset.input_length, debug_shapes=config['debug_shapes'])
    print(f"Model created with dynamic input length: {dataset.input_length}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Save architecture info
    with open(os.path.join(output_dir, 'architecture.json'), 'w') as f:
        json.dump(model.architecture_info, f, indent=4)
    
    # Create trainer
    trainer = VAETrainer(model, model_config)
    
    # Train model
    trainer.train(train_loader, val_loader, output_dir)
    
    # Plot training history
    trainer.plot_training_history(output_dir)
    
    # Plot reconstructions
    trainer.plot_reconstructions(val_loader, output_dir)
    
    # Plot latent space
    trainer.plot_latent_space(val_loader, output_dir)
    
    # Generate samples
    samples = trainer.generate_samples(num_samples=10)
    print(f"Generated {len(samples)} samples from VAE")
    
    # Save training history
    training_history = {
        'train_losses': trainer.train_losses,
        'val_losses': trainer.val_losses,
        'recon_losses': trainer.recon_losses,
        'kl_losses': trainer.kl_losses,
        'beta_values': trainer.beta_values,
        'best_val_loss': trainer.best_val_loss
    }
    
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(training_history, f, indent=4)
    
    print(f"Results saved to: {output_dir}")
    return trainer

def main():
    """Main training function"""
    # Create base output directory
    os.makedirs(CONFIG['base_output_dir'], exist_ok=True)
    
    # Save main configuration
    with open(os.path.join(CONFIG['base_output_dir'], 'config.json'), 'w') as f:
        json.dump(CONFIG, f, indent=4)
    
    # Create dataset
    dataset = VAEDataset(CONFIG['LABEL_CSV'], CONFIG['SPECTRA_DIR'], CONFIG['n_spectra'])
    
    # Train models for each latent dimension
    trainers = {}
    for latent_dim in CONFIG['latent_dims']:
        trainer = train_vae_for_latent_dim(latent_dim, CONFIG, dataset)
        trainers[latent_dim] = trainer
    
    print(f"\n=== All VAE Models Trained ===")
    print(f"Trained {len(CONFIG['latent_dims'])} models with latent dimensions: {CONFIG['latent_dims']}")
    print(f"Results saved to: {CONFIG['base_output_dir']}")

if __name__ == "__main__":
    main()
