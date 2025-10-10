#!/usr/bin/env python3
"""
Improved Variational Autoencoder for Stellar Spectroscopy
Addresses the "too sharp" training issue with better learning dynamics,
learning rate scheduling, and comprehensive logging for model comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
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

# Improved Configuration
CONFIG = {
    'input_length': 16384,
    'latent_dims': [16, 32, 64, 128, 256, 512, 1024],  # Multiple latent dimensions to try
    'channels': '32,64,128,256',  # More gradual channel progression
    'kernels': '64,32,16,8',      # More gradual kernel progression
    'pools_or_strides': '4,4,4,4', # Smaller strides for more gradual downsampling
    'downsampling_mode': 'stride',
    'learning_rate': 5e-5,         # Lower learning rate for more stable training
    'batch_size': 16,              # Smaller batch size for better gradients
    'num_epochs': 150,             # More epochs for gradual learning
    'beta': 0.1,                   # Lower beta to start (KL annealing)
    'beta_final': 1.0,             # Final beta value
    'beta_annealing_epochs': 50,   # Epochs to anneal beta
    'early_stopping_patience': 25, # More patience
    'base_output_dir': './results/variational_autoencoder_improved',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'LABEL_CSV': './galah_data/metadata.csv',
    'SPECTRA_DIR': './galah_data/fits',
    'debug_shapes': False,         # Set to False for cleaner output
    'use_lr_scheduler': True,      # Enable learning rate scheduling
    'use_gradient_clipping': True, # Enable gradient clipping
    'gradient_clip_value': 1.0,    # Gradient clipping threshold
    'weight_decay': 1e-5,          # L2 regularization
    'dropout_rate': 0.2            # Dropout for regularization
}

class ImprovedVAEDataset(Dataset):
    """Improved dataset with better preprocessing"""
    
    def __init__(self, csv_path, spectra_dir, input_length=16384):
        self.csv_path = csv_path
        self.spectra_dir = spectra_dir
        self.input_length = input_length
        
        # Load metadata
        self.df = pd.read_csv(csv_path)
        print(f"Dataset size: {len(self.df)}")
        
        # Filter for available spectra
        self.available_files = []
        for idx, row in self.df.iterrows():
            sobject_id = row['sobject_id']
            # Check if any of the FITS files exist
            found = False
            for cam in [1, 2, 3, 4]:
                fpath = os.path.join(spectra_dir, f"{sobject_id}{cam}.fits")
                if os.path.exists(fpath):
                    self.available_files.append(idx)
                    found = True
                    break
        
        print(f"Available spectra: {len(self.available_files)}")
        
        # Create common wavelength grid (adjust range as needed)
        self.common_wavelength = np.linspace(4000, 7000, input_length)
        
    def __len__(self):
        return len(self.available_files)
    
    def get_flux_from_all_cameras(self, sobject_id):
        """Extract flux from all cameras and interpolate to common grid"""
        try:
            all_wavelengths, all_fluxes = [], []
            
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
                return None, None
            
            # Concatenate all data
            full_wave = np.concatenate(all_wavelengths)
            full_flux = np.concatenate(all_fluxes)
            sort_idx = np.argsort(full_wave)
            full_wave = full_wave[sort_idx]
            full_flux = full_flux[sort_idx]
            
            # Interpolate to common grid
            interp_func = interp1d(full_wave, full_flux, kind='linear', 
                                  bounds_error=False, fill_value=0.0)
            interp_flux = interp_func(self.common_wavelength)
            
            # Normalize the flux
            if np.std(interp_flux) > 0:
                interp_flux = (interp_flux - np.mean(interp_flux)) / np.std(interp_flux)
            
            return self.common_wavelength, interp_flux
            
        except Exception as e:
            return None, None
    
    def __getitem__(self, idx):
        row_idx = self.available_files[idx]
        sobject_id = self.df.iloc[row_idx]['sobject_id']
        
        wave, flux = self.get_flux_from_all_cameras(sobject_id)
        
        if flux is None:
            # Return zeros if loading failed
            return torch.zeros(1, self.input_length, dtype=torch.float32)
        
        return torch.tensor(flux, dtype=torch.float32).unsqueeze(0)

class ImprovedVAE(nn.Module):
    """Improved VAE with better architecture and training dynamics"""
    
    def __init__(self, config, debug_shapes=False):
        super().__init__()
        
        # Store config for access in methods
        self.config = config
        
        # Parse configuration
        self.input_length = config['input_length']
        self.latent_dim = config['latent_dim']
        self.channels = [int(c) for c in config['channels'].split(',')]
        self.kernels = [int(k) for k in config['kernels'].split(',')]
        self.pools_or_strides = [int(s) for s in config['pools_or_strides'].split(',')]
        self.downsampling_mode = config['downsampling_mode']
        
        # Encoder
        self.encoder_layers = nn.ModuleList()
        in_channels = 1
        current_length = self.input_length
        
        for i, (c, k, s) in enumerate(zip(self.channels, self.kernels, self.pools_or_strides)):
            # Convolutional layer
            padding = (k - 1) // 2
            self.encoder_layers.append(
                nn.Conv1d(in_channels, c, kernel_size=k, stride=s, padding=padding)
            )
            
            # Batch normalization
            self.encoder_layers.append(nn.BatchNorm1d(c))
            
            # Activation with dropout
            self.encoder_layers.append(nn.ReLU())
            if config['dropout_rate'] > 0:
                self.encoder_layers.append(nn.Dropout(config['dropout_rate']))
            
            # Calculate output size
            current_length = (current_length + 2 * padding - k) // s + 1
            in_channels = c
            
            if debug_shapes:
                print(f"Encoder layer {i}: {current_length} (channels: {c}, kernel: {k}, stride: {s})")
        
        # Calculate flattened size
        self.flattened_size = in_channels * current_length
        
        # Latent space
        self.fc_mu = nn.Linear(self.flattened_size, self.latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, self.latent_dim)
        
        # Decoder
        self.fc_decoder = nn.Linear(self.latent_dim, self.flattened_size)
        
        # Reverse the architecture for decoder
        rev_channels = list(reversed(self.channels))
        rev_kernels = list(reversed(self.kernels))
        rev_pools_or_strides = list(reversed(self.pools_or_strides))
        
        self.decoder_layers = nn.ModuleList()
        
        for i in range(len(rev_channels)):
            if i == 0:
                # First decoder layer: input from latent space
                in_ch = rev_channels[i]  # 256
                out_ch = rev_channels[i]  # 256 (keep same for first layer)
            elif i == len(rev_channels) - 1:
                # Last decoder layer: output to final spectrum
                in_ch = rev_channels[i-1]  # 32
                out_ch = 1  # Final output (1 channel)
            else:
                # Middle layers: connect adjacent channels
                in_ch = rev_channels[i-1]  # Previous layer's output
                out_ch = rev_channels[i]    # Current layer's output
            
            k = rev_kernels[i]
            s = rev_pools_or_strides[i]
            padding = (k - 1) // 2
            
            # Transpose convolution
            self.decoder_layers.append(
                nn.ConvTranspose1d(in_ch, out_ch, kernel_size=k, stride=s, padding=padding)
            )
            
            if i < len(rev_channels) - 1:
                self.decoder_layers.append(nn.BatchNorm1d(out_ch))
                self.decoder_layers.append(nn.ReLU())
                if config['dropout_rate'] > 0:
                    self.decoder_layers.append(nn.Dropout(config['dropout_rate']))
        
        # Store architecture info
        self.architecture_info = {
            'input_length': self.input_length,
            'latent_dim': self.latent_dim,
            'channels': self.channels,
            'kernels': self.kernels,
            'pools_or_strides': self.pools_or_strides,
            'downsampling_mode': self.downsampling_mode,
            'flattened_size': self.flattened_size,
            'total_parameters': sum(p.numel() for p in self.parameters())
        }
    
    def encode(self, x):
        """Encode input to latent parameters"""
        for layer in self.encoder_layers:
            x = layer(x)
        
        x = x.flatten(start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent vector to output"""
        x = self.fc_decoder(z)
        x = x.view(x.size(0), self.channels[-1], -1)
        
        for layer in self.decoder_layers:
            x = layer(x)
        
        return x
    
    def forward(self, x):
        """Forward pass"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

class ImprovedVAETrainer:
    """Improved trainer with better training dynamics"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config['device']
        self.model.to(self.device)
        
        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        if config['use_lr_scheduler']:
            self.scheduler = CosineAnnealingLR(
                self.optimizer, 
                T_max=config['num_epochs'],
                eta_min=config['learning_rate'] * 0.01
            )
        else:
            self.scheduler = None
        
        # Loss function
        self.reconstruction_loss = nn.MSELoss()
        
        # Training history with more detailed tracking
        self.train_losses = []
        self.val_losses = []
        self.recon_losses = []
        self.kl_losses = []
        self.learning_rates = []
        self.beta_values = []
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def get_beta(self, epoch):
        """Get beta value for KL annealing"""
        if epoch < self.config['beta_annealing_epochs']:
            # Linear annealing from 0 to beta_final
            beta = self.config['beta'] + (self.config['beta_final'] - self.config['beta']) * epoch / self.config['beta_annealing_epochs']
        else:
            beta = self.config['beta_final']
        return beta
    
    def loss_function(self, recon_x, x, mu, logvar, beta=1.0):
        """Improved VAE loss function"""
        # Handle size mismatch
        min_size = min(recon_x.size(-1), x.size(-1))
        recon_x_trimmed = recon_x[..., :min_size]
        x_trimmed = x[..., :min_size]
        
        # Reconstruction loss
        recon_loss = self.reconstruction_loss(recon_x_trimmed, x_trimmed)
        
        # KL divergence with better numerical stability
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def train_epoch(self, train_loader, epoch):
        """Train one epoch with improved dynamics"""
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        beta = self.get_beta(epoch)
        
        for batch_idx, data in enumerate(train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            
            # Forward pass
            recon_batch, mu, logvar = self.model(data)
            
            # Calculate loss
            loss, recon_loss, kl_loss = self.loss_function(recon_batch, data, mu, logvar, beta)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['use_gradient_clipping']:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip_value'])
            
            self.optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            
            # Print progress every 20 batches
            if batch_idx % 20 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}, Beta: {beta:.3f}")
        
        return (total_loss / len(train_loader), 
                total_recon_loss / len(train_loader), 
                total_kl_loss / len(train_loader),
                beta)
    
    def validate(self, val_loader, epoch):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        beta = self.get_beta(epoch)
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(self.device)
                recon_batch, mu, logvar = self.model(data)
                loss, recon_loss, kl_loss = self.loss_function(recon_batch, data, mu, logvar, beta)
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
        
        return (total_loss / len(val_loader), 
                total_recon_loss / len(val_loader), 
                total_kl_loss / len(val_loader),
                beta)
    
    def train(self, train_loader, val_loader, output_dir):
        """Train the model with improved dynamics"""
        print(f"=== Training Improved VAE with Latent Dim {self.model.latent_dim} ===")
        print(f"Learning rate: {self.config['learning_rate']}")
        print(f"Beta annealing: {self.config['beta']} → {self.config['beta_final']} over {self.config['beta_annealing_epochs']} epochs")
        
        for epoch in range(self.config['num_epochs']):
            # Train
            train_loss, train_recon, train_kl, train_beta = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_recon, val_kl, val_beta = self.validate(val_loader, epoch)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.config['learning_rate']
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.recon_losses.append(train_recon)
            self.kl_losses.append(train_kl)
            self.learning_rates.append(current_lr)
            self.beta_values.append(train_beta)
            
            # Print progress every 5 epochs
            if (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{self.config["num_epochs"]}]')
                print(f'  Train Loss: {train_loss:.6f} (Recon: {train_recon:.6f}, KL: {train_kl:.6f})')
                print(f'  Val Loss: {val_loss:.6f} (Recon: {val_recon:.6f}, KL: {val_kl:.6f})')
                print(f'  Beta: {train_beta:.3f}, LR: {current_lr:.2e}')
                print('-' * 60)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 
                          os.path.join(output_dir, 'best_model.pth'))
                print(f"  ✅ New best model saved! Val Loss: {val_loss:.6f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config['early_stopping_patience']:
                    print(f"  🛑 Early stopping at epoch {epoch+1}")
                    break
        
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
    
    def plot_training_history(self, output_dir):
        """Create comprehensive training history plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Improved VAE Training History (L_{self.model.latent_dim})', fontsize=16)
        
        # 1. Total loss
        axes[0, 0].plot(self.train_losses, label='Train Loss', linewidth=2)
        axes[0, 0].plot(self.val_losses, label='Val Loss', linewidth=2)
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
        
        # 2. Reconstruction loss
        axes[0, 1].plot(self.recon_losses, label='Reconstruction Loss', linewidth=2)
        axes[0, 1].set_title('Reconstruction Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')
        
        # 3. KL divergence
        axes[0, 2].plot(self.kl_losses, label='KL Divergence', color='red', linewidth=2)
        axes[0, 2].set_title('KL Divergence')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_yscale('log')
        
        # 4. Learning rate
        axes[1, 0].plot(self.learning_rates, label='Learning Rate', color='green', linewidth=2)
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
        
        # 5. Beta values
        axes[1, 1].plot(self.beta_values, label='Beta (KL Weight)', color='purple', linewidth=2)
        axes[1, 1].set_title('Beta Annealing')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Beta')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Loss components comparison
        axes[1, 2].plot(self.recon_losses, label='Reconstruction', alpha=0.7)
        axes[1, 2].plot(self.kl_losses, label='KL Divergence', alpha=0.7)
        axes[1, 2].set_title('Loss Components')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Loss')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_history_comprehensive.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also save the simple version for comparison
        self.plot_simple_history(output_dir)
    
    def plot_simple_history(self, output_dir):
        """Create simple training history plot for comparison"""
        plt.figure(figsize=(12, 8))
        
        # Total loss
        plt.subplot(2, 2, 1)
        plt.plot(self.train_losses, label='Train Loss', linewidth=2)
        plt.plot(self.val_losses, label='Val Loss', linewidth=2)
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Reconstruction loss
        plt.subplot(2, 2, 2)
        plt.plot(self.recon_losses, label='Reconstruction Loss', linewidth=2)
        plt.title('Reconstruction Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # KL divergence
        plt.subplot(2, 2, 3)
        plt.plot(self.kl_losses, label='KL Divergence', color='red', linewidth=2)
        plt.title('KL Divergence')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Learning rate
        plt.subplot(2, 2, 4)
        plt.plot(self.learning_rates, label='Learning Rate', color='green', linewidth=2)
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_latent_space(self, test_loader, output_dir, num_samples=1000):
        """Plot latent space representation with fixed Unicode characters"""
        print("Plotting latent space representation...")
        
        self.model.eval()
        mus = []
        logvars = []
        
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                if i * self.config['batch_size'] >= num_samples:
                    break
                
                # Handle both tuple and single tensor formats
                if isinstance(data, tuple):
                    spectra = data[0]  # If it's a tuple (spectra, _)
                else:
                    spectra = data      # If it's just the spectra tensor
                
                spectra = spectra.to(self.device)
                mu, logvar = self.model.encode(spectra)
                mus.append(mu.cpu())
                logvars.append(logvar.cpu())
        
        mus = torch.cat(mus, dim=0)[:num_samples]
        logvars = torch.cat(logvars, dim=0)[:num_samples]
        
        # Convert to numpy for plotting
        mus = mus.numpy()
        logvars = logvars.numpy()
        
        # Create latent space plots
        plt.figure(figsize=(15, 6))
        
        # Plot mu (mean) of latent space
        plt.subplot(1, 2, 1)
        plt.scatter(mus[:, 0], mus[:, 1], alpha=0.6)
        plt.title(f'Latent Space (mu) - First 2 Dimensions (L_{self.model.latent_dim})')
        plt.xlabel('mu_1')
        plt.ylabel('mu_2')
        plt.grid(True)
        
        # Plot log variance of latent space
        plt.subplot(1, 2, 2)
        plt.scatter(logvars[:, 0], logvars[:, 1], alpha=0.6)
        plt.title(f'Latent Space (log variance) - First 2 Dimensions (L_{self.model.latent_dim})')
        plt.xlabel('log variance_1')
        plt.ylabel('log variance_2')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'latent_space.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Latent space plot saved to {output_dir}")
        
        # Also save the latent representations for further analysis
        latent_data = {
            'mus': mus.tolist(),
            'logvars': logvars.tolist(),
            'latent_dim': self.model.latent_dim,
            'num_samples': len(mus)
        }
        
        with open(os.path.join(output_dir, 'latent_representations.json'), 'w') as f:
            json.dump(latent_data, f, indent=4)
        
        print(f"✅ Latent representations saved to {output_dir}")

def train_improved_vae_for_latent_dim(latent_dim, config, dataset):
    """Train improved VAE for a specific latent dimension"""
    
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
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    print(f"\n=== Training Improved VAE with Latent Dimension {latent_dim} ===")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create model
    model = ImprovedVAE(model_config, debug_shapes=config['debug_shapes'])
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Save architecture info
    with open(os.path.join(output_dir, 'architecture.json'), 'w') as f:
        json.dump(model.architecture_info, f, indent=4)
    
    # Create trainer
    trainer = ImprovedVAETrainer(model, model_config)
    
    # Train model
    trainer.train(train_loader, val_loader, output_dir)
    
    # Plot training history
    trainer.plot_training_history(output_dir)
    
    # Plot latent space representation
    trainer.plot_latent_space(val_loader, output_dir)
    
    # Save training history
    training_history = {
        'train_losses': trainer.train_losses,
        'val_losses': trainer.val_losses,
        'recon_losses': trainer.recon_losses,
        'kl_losses': trainer.kl_losses,
        'learning_rates': trainer.learning_rates,
        'beta_values': trainer.beta_values,
        'best_val_loss': trainer.best_val_loss,
        'final_epoch': len(trainer.train_losses)
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
    dataset = ImprovedVAEDataset(CONFIG['LABEL_CSV'], CONFIG['SPECTRA_DIR'], CONFIG['input_length'])
    
    # Train models for each latent dimension
    trainers = {}
    for latent_dim in CONFIG['latent_dims']:
        trainer = train_improved_vae_for_latent_dim(latent_dim, CONFIG, dataset)
        trainers[latent_dim] = trainer
    
    print(f"\n=== All Improved VAE Models Trained ===")
    print(f"Trained {len(CONFIG['latent_dims'])} models with latent dimensions: {CONFIG['latent_dims']}")
    print(f"Results saved to: {CONFIG['base_output_dir']}")
    
    # Create comparison plot
    create_comparison_plot(trainers, CONFIG['base_output_dir'])

def create_comparison_plot(trainers, output_dir):
    """Create comparison plot of all models"""
    plt.figure(figsize=(15, 10))
    
    # Plot validation losses for all models
    for latent_dim, trainer in trainers.items():
        plt.plot(trainer.val_losses, label=f'L_{latent_dim}', linewidth=2, alpha=0.8)
    
    plt.title('Validation Loss Comparison - All Latent Dimensions')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_models_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Comparison plot saved to {output_dir}")

if __name__ == "__main__":
    main() 