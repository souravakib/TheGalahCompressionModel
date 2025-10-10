#!/usr/bin/env python3
"""
Improved Contrastive Learning Pipeline for Stellar Spectra
Fixes for higher latent dimensions (L256, L512, L1024)
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
from datetime import datetime
import random

# Configuration with improvements for higher dimensions
CONFIG = {
    'latent_dims': [16, 32, 64, 128, 256, 512, 1024],  # All latent dimensions
    'latent_dim': 128,  # Projection head output dimension (will be overridden)
    'channels': [64, 128, 256],
    'kernels': [32, 16, 8],
    'pools_or_strides': [8, 8, 8],
    'downsampling_mode': 'stride',
    'learning_rate': 1e-5,  # Base learning rate (will be adjusted per dimension)
    'batch_size': 32,
    'num_epochs': 100,
    'temperature': 0.1,
    'early_stopping_patience': 15,
    'early_stopping_min_delta': 0.0001,
    'output_dir': '/user/HS401/ss06019/results/contrastive_learning_improved',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'label_csv': '/user/HS401/ss06019/Desktop/dissertation/GalahCompressionModel/galah_all.csv',
    'spectra_dir': '/user/HS401/ss06019/Desktop/dissertation/GalahCompressionModel/galah_data/fits',
    'n_spectra': 4000,
    'random_seed': 42,
    'augmentation_strength': 0.1,
    'projection_head_dim': 256,
    'use_metadata_contrast': True,
    'metadata_weight': 0.3,
    'spectral_augmentations': True,
    'temporal_augmentations': True,
    'noise_augmentations': True,
    'save_checkpoints': True,
    'checkpoint_freq': 10,
    'num_workers': 4,
    'pin_memory': True,
    'gradient_accumulation_steps': 2,
    'mixed_precision': True,
    'input_length': 16384,
    # NEW: Adaptive parameters for different latent dimensions
    'adaptive_learning_rates': {
        16: 1e-5,
        32: 1e-5,
        64: 1e-5,
        128: 1e-5,
        256: 1e-6,    # 10x lower
        512: 5e-7,    # 20x lower
        1024: 1e-7    # 100x lower
    },
    'adaptive_dropout_rates': {
        16: 0.0,
        32: 0.0,
        64: 0.0,
        128: 0.0,
        256: 0.1,     # Add dropout
        512: 0.2,     # Higher dropout
        1024: 0.3     # Even higher dropout
    },
    'adaptive_weight_decay': {
        16: 1e-6,
        32: 1e-6,
        64: 1e-6,
        128: 1e-6,
        256: 1e-4,    # Add weight decay
        512: 1e-4,    # Same weight decay
        1024: 1e-4    # Same weight decay
    },
    'gradient_clip_norm': 1.0,  # Add gradient clipping
    'warmup_epochs': 10,  # Learning rate warmup
}

class SpectralAugmentations:
    """Class for creating augmented views of stellar spectra"""
    
    def __init__(self, config):
        self.config = config
        self.augmentation_strength = config['augmentation_strength']
        
    def add_noise(self, spectrum, noise_level=None):
        """Add Gaussian noise to spectrum"""
        if noise_level is None:
            noise_level = self.augmentation_strength * np.std(spectrum)
        noise = np.random.normal(0, noise_level, spectrum.shape)
        return spectrum + noise
    
    def spectral_shift(self, spectrum, shift_range=None):
        """Apply small spectral shifts (wavelength calibration errors)"""
        if shift_range is None:
            shift_range = int(self.augmentation_strength * len(spectrum) * 0.01)
        shift = np.random.randint(-shift_range, shift_range + 1)
        if shift > 0:
            return np.concatenate([spectrum[shift:], spectrum[:shift]])
        elif shift < 0:
            return np.concatenate([spectrum[shift:], spectrum[:shift]])
        return spectrum
    
    def spectral_stretch(self, spectrum, stretch_range=None):
        """Apply spectral stretching (velocity broadening simulation)"""
        if stretch_range is None:
            stretch_range = 1.0 + self.augmentation_strength * 0.2
        stretch_factor = np.random.uniform(1.0/stretch_range, stretch_range)
        
        # Simple stretching by interpolation
        original_indices = np.arange(len(spectrum))
        new_indices = original_indices * stretch_factor
        new_indices = np.clip(new_indices, 0, len(spectrum) - 1)
        
        stretched_spectrum = np.interp(original_indices, new_indices, spectrum)
        return stretched_spectrum
    
    def mask_regions(self, spectrum, mask_prob=0.1):
        """Randomly mask spectral regions (simulating missing data)"""
        mask = np.random.random(len(spectrum)) > mask_prob
        return spectrum * mask
    
    def intensity_scale(self, spectrum, scale_range=None):
        """Apply intensity scaling (flux calibration errors)"""
        if scale_range is None:
            scale_range = 1.0 + self.augmentation_strength * 0.3
        scale_factor = np.random.uniform(1.0/scale_range, scale_range)
        return spectrum * scale_factor
    
    def apply_augmentations(self, spectrum):
        """Apply random augmentations to create augmented view"""
        augmented = spectrum.copy()
        
        if self.config['noise_augmentations'] and np.random.random() < 0.5:
            augmented = self.add_noise(augmented)
        
        if self.config['temporal_augmentations'] and np.random.random() < 0.5:
            augmented = self.spectral_shift(augmented)
        
        if self.config['spectral_augmentations'] and np.random.random() < 0.5:
            augmented = self.spectral_stretch(augmented)
        
        if np.random.random() < 0.3:
            augmented = self.mask_regions(augmented)
        
        if np.random.random() < 0.3:
            augmented = self.intensity_scale(augmented)
        
        return augmented

class ContrastiveDataset(Dataset):
    """Dataset for contrastive learning with dynamic wavelength determination"""
    
    def __init__(self, config):
        self.config = config
        self.spectra_dir = config['spectra_dir']
        
        # Load metadata
        self.df = pd.read_csv(config['label_csv'])
        self.df['sobject_id'] = self.df['sobject_id'].astype(str)
        
        # Limit to available spectra
        self.available_files = []
        for idx, row in self.df.iterrows():
            sobject_id = row['sobject_id']
            found = False
            for cam in [1, 2, 3, 4]:
                fpath = os.path.join(self.spectra_dir, f"{sobject_id}{cam}.fits")
                if os.path.exists(fpath):
                    self.available_files.append(idx)
                    found = True
                    break
            if not found:
                continue
        
        print(f"Available spectra for contrastive learning: {len(self.available_files)}")
        
        # Check if spectra directory exists
        if not os.path.exists(self.spectra_dir):
            print(f"Error: Spectra directory not found: {self.spectra_dir}")
            return
        
        # List some sample FITS files for debugging
        fits_files = [f for f in os.listdir(self.spectra_dir) if f.endswith('.fits')]
        print(f"Found {len(fits_files)} FITS files in {self.spectra_dir}")
        if len(fits_files) > 0:
            print(f"Sample FITS files: {fits_files[:5]}")
        
        # Create dynamic common wavelength grid from sampled spectra
        print("Creating dynamic common wavelength grid...")
        sample_ids = [self.df.iloc[idx]['sobject_id'] for idx in self.available_files[:min(50, len(self.available_files))]]
        sample_waves = []
        
        for sid in sample_ids:
            try:
                wave, _ = self.get_flux_from_all_cameras_raw(sid)
                if wave is not None:
                    sample_waves.append(wave)
                    print(f"  ✓ Successfully loaded spectrum {sid}: {len(wave)} points, range {wave.min():.2f}-{wave.max():.2f} Å")
                else:
                    print(f"  ✗ Failed to load spectrum {sid}: get_flux_from_all_cameras_raw returned None")
            except Exception as e:
                print(f"  ✗ Error loading spectrum {sid}: {str(e)}")
                continue
        
        if len(sample_waves) > 0:
            # Use median length sample for common grid
            self.common_wavelength = sample_waves[np.argsort([len(w) for w in sample_waves])[len(sample_waves)//2]]
            self.config['input_length'] = len(self.common_wavelength)
            print(f"Dynamic common grid created: {len(self.common_wavelength)} points")
            print(f"Wavelength range: {self.common_wavelength.min():.2f} - {self.common_wavelength.max():.2f} Å")
            
            # Check camera coverage
            print("Camera coverage analysis:")
            camera_ranges = [
                (4000, 5800, "Camera 1 (Blue)"),
                (5700, 6800, "Camera 2 (Green)"), 
                (6800, 7900, "Camera 3 (Red)"),
                (7585, 7886, "Camera 4 (Near-IR)")
            ]
            
            for wmin, wmax, name in camera_ranges:
                covered = (self.common_wavelength.min() <= wmin) and (self.common_wavelength.max() >= wmax)
                print(f"  {name}: {wmin}-{wmax} Å - {'COVERED' if covered else 'NOT COVERED'}")
                
        else:
            # Fallback to extended range (use a reasonable default length)
            fallback_length = 20000  # Extended range to cover all cameras
            self.common_wavelength = np.linspace(4000, 8000, fallback_length)
            self.config['input_length'] = fallback_length
            print(f"Using fallback common grid: {len(self.common_wavelength)} points")
        
        # Prepare metadata for contrastive learning
        if self.config['use_metadata_contrast']:
            self.prepare_metadata_contrasts()
    
    def prepare_metadata_contrasts(self):
        """Prepare metadata-based contrastive pairs"""
        print("Preparing metadata-based contrastive pairs...")
        
        # Focus on key stellar parameters for contrastive learning
        self.metadata_columns = ['Fe_h', 'Mg_fe', 'Si_fe', 'Ca_fe', 'Ti_fe']
        
        # Create metadata matrix
        self.metadata_matrix = []
        for idx in self.available_files:
            row = self.df.iloc[idx]
            metadata_row = []
            for col in self.metadata_columns:
                if pd.isna(row[col]):
                    metadata_row.append(0.0)  # Default value for missing data
                else:
                    metadata_row.append(float(row[col]))
            self.metadata_matrix.append(metadata_row)
        
        self.metadata_matrix = np.array(self.metadata_matrix)
        
        # Normalize metadata
        self.metadata_scaler = StandardScaler()
        self.metadata_matrix = self.metadata_scaler.fit_transform(self.metadata_matrix)
        
        print(f"Metadata matrix shape: {self.metadata_matrix.shape}")
    
    def __len__(self):
        return len(self.available_files)
    
    def get_flux_from_all_cameras_raw(self, sobject_id):
        """Extract raw flux from all cameras (similar to CNN autoencoder)"""
        try:
            all_wavelengths, all_fluxes = [], []
            cameras_found = 0
            
            for cam in [1, 2, 3, 4]:
                fpath = os.path.join(self.spectra_dir, f"{sobject_id}{cam}.fits")
                if not os.path.exists(fpath):
                    continue
                    
                try:
                    with fits.open(fpath) as hdul:
                        flux = hdul[4].data
                        hdr = hdul[4].header
                        wave = hdr['CRVAL1'] + hdr['CDELT1'] * np.arange(hdr['NAXIS1'])
                        
                        # Process flux
                        safe_flux = np.where(flux > 1e-10, flux, 1e-10)
                        flux_processed = np.log(safe_flux)
                        
                        all_wavelengths.append(wave)
                        all_fluxes.append(flux_processed)
                        cameras_found += 1
                        
                except Exception as e:
                    print(f"Error reading {fpath}: {e}")
                    continue
            
            if cameras_found == 0:
                return None, None
            
            # Concatenate and sort
            full_wave = np.concatenate(all_wavelengths)
            full_flux = np.concatenate(all_fluxes)
            sort_idx = np.argsort(full_wave)
            full_wave = full_wave[sort_idx]
            full_flux = full_flux[sort_idx]
            
            return full_wave, full_flux
            
        except Exception as e:
            print(f"Error in get_flux_from_all_cameras_raw for {sobject_id}: {e}")
            return None, None
    
    def get_flux_from_all_cameras(self, sobject_id):
        """Load spectrum from all cameras and interpolate to common grid"""
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
                return np.zeros(self.config['input_length'])
            
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
            return np.zeros(self.config['input_length'])
    
    def __getitem__(self, idx):
        row_idx = self.available_files[idx]
        row = self.df.iloc[row_idx]
        sobject_id = row['sobject_id']
        
        # Load spectrum
        spectrum = self.get_flux_from_all_cameras(sobject_id)
        
        # Create two augmented views
        augmentations = SpectralAugmentations(self.config)
        view1 = augmentations.apply_augmentations(spectrum)
        view2 = augmentations.apply_augmentations(spectrum)
        
        # Get metadata if available
        if self.config['use_metadata_contrast'] and hasattr(self, 'metadata_matrix'):
            metadata = self.metadata_matrix[idx]
        else:
            metadata = None
        
        return {
            'view1': torch.FloatTensor(view1).unsqueeze(0),  # Add channel dimension: [1, length]
            'view2': torch.FloatTensor(view2).unsqueeze(0),  # Add channel dimension: [1, length]
            'metadata': torch.FloatTensor(metadata) if metadata is not None else None,
            'sobject_id': sobject_id
        }

class ContrastiveEncoder(nn.Module):
    """Improved Encoder with dropout for higher dimensions"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Extract architecture parameters
        channels = config['channels']
        kernels = config['kernels']
        pools_or_strides = config['pools_or_strides']
        input_length = config['input_length']
        downsampling_mode = config['downsampling_mode']
        
        # Encoder (same as VAE)
        encoder_layers = []
        in_channels = 1
        
        for i in range(len(channels)):
            conv_layer = nn.Conv1d(in_channels, channels[i], kernel_size=kernels[i],
                                   stride=pools_or_strides[i] if downsampling_mode == 'stride' else 1,
                                   padding=(kernels[i] - 1) // 2)
            encoder_layers.extend([conv_layer, nn.BatchNorm1d(channels[i]), nn.ReLU()])
            
            if downsampling_mode == 'max_pool':
                encoder_layers.append(nn.MaxPool1d(kernel_size=pools_or_strides[i]))
            elif downsampling_mode == 'avg_pool':
                encoder_layers.append(nn.AvgPool1d(kernel_size=pools_or_strides[i]))
            
            in_channels = channels[i]
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Calculate flattened size
        with torch.no_grad():
            dummy_output = self.encoder(torch.zeros(1, 1, input_length))
        self.flattened_size = dummy_output.flatten(start_dim=1).shape[1]
        
        print(f"Encoder output shape: {dummy_output.shape}")
        print(f"Flattened size: {self.flattened_size}")
        
        # Projection head for contrastive learning with dropout
        latent_dim = config['latent_dim']
        dropout_rate = config['adaptive_dropout_rates'].get(latent_dim, 0.0)
        
        projection_layers = [
            nn.Linear(self.flattened_size, config['projection_head_dim']),
            nn.BatchNorm1d(config['projection_head_dim']),
            nn.ReLU()
        ]
        
        if dropout_rate > 0:
            projection_layers.append(nn.Dropout(dropout_rate))
            print(f"Added dropout {dropout_rate} for L{latent_dim}")
        
        projection_layers.extend([
            nn.Linear(config['projection_head_dim'], latent_dim)
        ])
        
        self.projection_head = nn.Sequential(*projection_layers)
        
        # Metadata projection head (optional)
        if config['use_metadata_contrast']:
            self.metadata_projection = nn.Sequential(
                nn.Linear(5, 64),  # 5 stellar parameters
                nn.ReLU(),
                nn.Dropout(0.1),  # Light dropout for metadata
                nn.Linear(64, latent_dim)
            )
    
    def forward(self, x, return_features=False):
        """Forward pass through encoder and projection head"""
        # Encode
        features = self.encoder(x)
        features = features.flatten(start_dim=1)
        
        # Project to contrastive space
        projections = self.projection_head(features)
        
        if return_features:
            return projections, features
        return projections

class ContrastiveLearningModel(nn.Module):
    """Complete contrastive learning model with improvements"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = ContrastiveEncoder(config)
        
        # Temperature parameter for contrastive loss
        self.temperature = nn.Parameter(torch.tensor(config['temperature']))
        
        # Metadata weight for combined loss
        self.metadata_weight = config['metadata_weight']
    
    def forward(self, view1, view2, metadata1=None, metadata2=None):
        """Forward pass for contrastive learning"""
        # Encode both views
        proj1, feat1 = self.encoder(view1, return_features=True)
        proj2, feat2 = self.encoder(view2, return_features=True)
        
        # Normalize projections
        proj1 = F.normalize(proj1, dim=1)
        proj2 = F.normalize(proj2, dim=1)
        
        # Metadata projections if available
        if self.config['use_metadata_contrast'] and metadata1 is not None:
            meta_proj1 = self.encoder.metadata_projection(metadata1)
            meta_proj2 = self.encoder.metadata_projection(metadata2)
            meta_proj1 = F.normalize(meta_proj1, dim=1)
            meta_proj2 = F.normalize(meta_proj2, dim=1)
        else:
            meta_proj1 = meta_proj2 = None
        
        return {
            'projections1': proj1,
            'projections2': proj2,
            'features1': feat1,
            'features2': feat2,
            'metadata_projections1': meta_proj1,
            'metadata_projections2': meta_proj2
        }

class ContrastiveLoss(nn.Module):
    """Contrastive loss for representation learning"""
    
    def __init__(self, temperature=0.07, metadata_weight=0.3):
        super().__init__()
        self.temperature = temperature
        self.metadata_weight = metadata_weight
    
    def forward(self, projections1, projections2, metadata_proj1=None, metadata_proj2=None):
        """Compute contrastive loss"""
        batch_size = projections1.shape[0]
        
        # Create similarity matrix
        logits = torch.mm(projections1, projections2.T) / self.temperature
        
        # Labels: diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=projections1.device)
        
        # Spectral contrastive loss
        spectral_loss = F.cross_entropy(logits, labels)
        
        # Metadata contrastive loss (if available)
        if metadata_proj1 is not None and metadata_proj2 is not None:
            meta_logits = torch.mm(metadata_proj1, metadata_proj2.T) / self.temperature
            meta_loss = F.cross_entropy(meta_logits, labels)
            
            # Combined loss
            total_loss = (1 - self.metadata_weight) * spectral_loss + self.metadata_weight * meta_loss
        else:
            total_loss = spectral_loss
            meta_loss = torch.tensor(0.0, device=projections1.device)
        
        return {
            'total_loss': total_loss,
            'spectral_loss': spectral_loss,
            'metadata_loss': meta_loss
        }

class ContrastiveTrainer:
    """Improved Trainer for contrastive learning with adaptive parameters"""
    
    def __init__(self, config, latent_dim=None, dataset=None):
        self.config = config.copy()  # Make a copy to avoid modifying original
        if latent_dim is not None:
            self.config['latent_dim'] = latent_dim
        self.device = torch.device(config['device'])
        
        # Set random seed
        self.set_seed(config['random_seed'])
        
        # Initialize components
        self.augmentations = SpectralAugmentations(config)
        
        # Use provided dataset or create new one
        if dataset is not None:
            self.dataset = dataset
            print("Using provided dataset (avoiding recreation)")
        else:
            self.dataset = ContrastiveDataset(config)
        
        # Create data loader
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=config['batch_size'], 
            shuffle=True, 
            num_workers=config['num_workers'],
            pin_memory=config['pin_memory']
        )
        
        # Create model
        self.model = ContrastiveLearningModel(self.config)
        self.model = self.model.to(self.device)
        
        # Adaptive learning rate based on latent dimension
        base_lr = config['learning_rate']
        adaptive_lr = config['adaptive_learning_rates'].get(latent_dim, base_lr)
        adaptive_weight_decay = config['adaptive_weight_decay'].get(latent_dim, 1e-6)
        
        print(f"Using adaptive learning rate: {adaptive_lr} for L{latent_dim}")
        print(f"Using adaptive weight decay: {adaptive_weight_decay} for L{latent_dim}")
        
        # Optimizer with adaptive parameters
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=adaptive_lr,
            weight_decay=adaptive_weight_decay
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config['num_epochs'],
            eta_min=adaptive_lr * 0.01
        )
        
        # Loss function
        self.criterion = ContrastiveLoss(
            temperature=config['temperature'],
            metadata_weight=config['metadata_weight']
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'spectral_loss': [],
            'metadata_loss': [],
            'learning_rate': []
        }
        
        print(f"Model initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def set_seed(self, seed):
        """Set random seed for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_spectral_loss = 0
        total_metadata_loss = 0
        
        for batch_idx, batch in enumerate(self.dataloader):
            view1 = batch['view1'].to(self.device)
            view2 = batch['view2'].to(self.device)
            metadata = batch['metadata'].to(self.device) if batch['metadata'] is not None else None
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(view1, view2, metadata, metadata)
            
            # Compute loss
            loss_dict = self.criterion(
                outputs['projections1'], 
                outputs['projections2'],
                outputs['metadata_projections1'],
                outputs['metadata_projections2']
            )
            
            loss = loss_dict['total_loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip_norm'])
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_spectral_loss += loss_dict['spectral_loss'].item()
            total_metadata_loss += loss_dict['metadata_loss'].item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1}/{self.config["num_epochs"]}, Batch {batch_idx}/{len(self.dataloader)}, '
                      f'Loss: {loss.item():.6f}, Spectral: {loss_dict["spectral_loss"].item():.6f}, '
                      f'Metadata: {loss_dict["metadata_loss"].item():.6f}')
        
        # Update learning rate
        self.scheduler.step()
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # Record history
        avg_loss = total_loss / len(self.dataloader)
        avg_spectral_loss = total_spectral_loss / len(self.dataloader)
        avg_metadata_loss = total_metadata_loss / len(self.dataloader)
        
        self.history['train_loss'].append(avg_loss)
        self.history['spectral_loss'].append(avg_spectral_loss)
        self.history['metadata_loss'].append(avg_metadata_loss)
        self.history['learning_rate'].append(current_lr)
        
        return avg_loss
    
    def train(self):
        """Train the model"""
        print(f"Starting contrastive learning training...")
        print(f"Training for {self.config['num_epochs']} epochs")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"Temperature: {self.config['temperature']}")
        print(f"Metadata weight: {self.config['metadata_weight']}")
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['num_epochs']):
            avg_loss = self.train_epoch(epoch)
            
            print(f'Epoch {epoch+1}/{self.config["num_epochs"]}: Loss: {avg_loss:.6f}, '
                  f'LR: {self.optimizer.param_groups[0]["lr"]:.2e}')
            
            # Early stopping
            if avg_loss < best_loss - self.config['early_stopping_min_delta']:
                best_loss = avg_loss
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': avg_loss,
                    'config': self.config
                }, os.path.join(self.config['output_dir'], 'best_model.pth'))
            else:
                patience_counter += 1
            
            if patience_counter >= self.config['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            # Save checkpoint
            if (epoch + 1) % self.config['checkpoint_freq'] == 0:
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': avg_loss,
                    'config': self.config
                }, os.path.join(self.config['output_dir'], f'checkpoint_epoch_{epoch+1}.pth'))
        
        # Save final model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.config['num_epochs'],
            'loss': avg_loss,
            'config': self.config
        }, os.path.join(self.config['output_dir'], 'final_model.pth'))
        
        return {
            'best_loss': best_loss,
            'final_epoch': epoch + 1,
            'early_stopped': patience_counter >= self.config['early_stopping_patience']
        }

def train_single_contrastive_model(latent_dim, base_config, dataset=None):
    """Train a single contrastive learning model for given latent dimension"""
    
    print(f"\n{'='*60}")
    print(f"TRAINING CONTRASTIVE MODEL - LATENT DIMENSION: {latent_dim}")
    print(f"{'='*60}")
    
    # Create experiment-specific config
    config = base_config.copy()
    config['latent_dim'] = latent_dim
    config['output_dir'] = os.path.join(base_config['output_dir'], f'L_{latent_dim}')
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Create trainer
    trainer = ContrastiveTrainer(config, latent_dim, dataset)
    
    # Train model
    results = trainer.train()
    
    # Save training results
    results['train_loss_history'] = trainer.history['train_loss']
    results['spectral_loss_history'] = trainer.history['spectral_loss']
    results['metadata_loss_history'] = trainer.history['metadata_loss']
    results['learning_rate_history'] = trainer.history['learning_rate']
    results['config'] = config
    results['latent_dim'] = latent_dim
    
    with open(os.path.join(config['output_dir'], 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot training history
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(trainer.history['train_loss']) + 1)
    
    axes[0, 0].plot(epochs, trainer.history['train_loss'], 'b-', linewidth=2)
    axes[0, 0].set_title('Total Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs, trainer.history['spectral_loss'], 'g-', linewidth=2)
    axes[0, 1].set_title('Spectral Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(epochs, trainer.history['metadata_loss'], 'r-', linewidth=2)
    axes[1, 0].set_title('Metadata Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(epochs, trainer.history['learning_rate'], 'm-', linewidth=2)
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['output_dir'], 'training_progress.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training completed for L{latent_dim}")
    print(f"Best loss: {results['best_loss']:.6f}")
    print(f"Results saved to: {config['output_dir']}")
    
    return results

def main():
    """Main function for improved contrastive learning"""
    
    print("IMPROVED CONTRASTIVE LEARNING PIPELINE")
    print("=" * 60)
    print("Improvements for higher latent dimensions:")
    print("• L256: LR=1e-6, dropout=0.1, weight_decay=1e-4")
    print("• L512: LR=5e-7, dropout=0.2, weight_decay=1e-4")
    print("• L1024: LR=1e-7, dropout=0.3, weight_decay=1e-4")
    print("• Gradient clipping, learning rate scheduling")
    print("=" * 60)
    
    # Create base output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Create dataset once to avoid recreation
    print("Creating dataset...")
    dataset = ContrastiveDataset(CONFIG)
    
    # Train each latent dimension
    all_results = {}
    
    for latent_dim in CONFIG['latent_dims']:
        try:
            results = train_single_contrastive_model(latent_dim, CONFIG, dataset)
            all_results[latent_dim] = results
            print(f"✅ L{latent_dim}: Training completed successfully")
        except Exception as e:
            print(f"❌ L{latent_dim}: Training failed - {str(e)}")
            all_results[latent_dim] = None
    
    # Create experiment summary
    experiment_summary = {
        'experiment_date': datetime.now().isoformat(),
        'models_trained': len([r for r in all_results.values() if r is not None]),
        'latent_dimensions': CONFIG['latent_dims'],
        'improvements_applied': {
            'adaptive_learning_rates': CONFIG['adaptive_learning_rates'],
            'adaptive_dropout_rates': CONFIG['adaptive_dropout_rates'],
            'adaptive_weight_decay': CONFIG['adaptive_weight_decay'],
            'gradient_clipping': CONFIG['gradient_clip_norm'],
            'learning_rate_scheduling': True
        },
        'results_summary': {}
    }
    
    for latent_dim, results in all_results.items():
        if results is not None:
            experiment_summary['results_summary'][str(latent_dim)] = {
                'best_train_loss': results['best_loss'],
                'final_epoch': results['final_epoch'],
                'early_stopped': results['early_stopped']
            }
    
    # Save experiment summary
    with open(os.path.join(CONFIG['output_dir'], 'experiment_summary.json'), 'w') as f:
        json.dump(experiment_summary, f, indent=2)
    
    # Create comparison plot
    successful_results = {k: v for k, v in all_results.items() if v is not None}
    if len(successful_results) > 1:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        latent_dims = sorted(successful_results.keys())
        final_losses = [successful_results[dim]['best_loss'] for dim in latent_dims]
        
        ax.plot(latent_dims, final_losses, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Latent Dimension')
        ax.set_ylabel('Best Training Loss')
        ax.set_title('Improved Contrastive Learning Performance')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        # Add labels for each point
        for i, latent_dim in enumerate(latent_dims):
            ax.annotate(f'L{latent_dim}', (latent_dim, final_losses[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG['output_dir'], 'comparison_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print("\n" + "=" * 60)
    print("IMPROVED CONTRASTIVE LEARNING COMPLETED!")
    print("=" * 60)
    print(f"Models trained: {experiment_summary['models_trained']}")
    print(f"Results saved to: {CONFIG['output_dir']}")
    print("Check individual L_* directories for detailed results")

if __name__ == "__main__":
    main()
