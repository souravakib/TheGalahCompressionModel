#!/usr/bin/env python3
"""
Contrastive Learning Pipeline for Stellar Spectra
Uses the VAE encoder architecture for representation learning
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

# Configuration
CONFIG = {
    'latent_dims': [16, 32, 64, 128, 256, 512, 1024],  # Multiple latent dimensions to train
    'latent_dim': 128,  # Projection head output dimension (will be overridden)
    'channels': [64, 128, 256],
    'kernels': [32, 16, 8],
    'pools_or_strides': [8, 8, 8],
    'downsampling_mode': 'stride',
    'learning_rate': 1e-5, 
    'batch_size': 32,  # Increased for lab GPU
    'num_epochs': 100,
    'temperature': 0.1,  
    'early_stopping_patience': 15,
    'early_stopping_min_delta': 0.0001,  
    'output_dir': '/user/HS401/ss06019/results/contrastive_learning',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'label_csv': '/user/HS401/ss06019/Desktop/dissertation/GalahCompressionModel/galah_all.csv',
    'spectra_dir': '/user/HS401/ss06019/Desktop/dissertation/GalahCompressionModel/galah_data/fits',
    'n_spectra': 4000,  # Increased for lab - more data = better representations
    'random_seed': 42,
    'augmentation_strength': 0.1, 
    'projection_head_dim': 256,  # Hidden dimension for projection head
    'use_metadata_contrast': True,  # Use stellar parameters for contrastive learning
    'metadata_weight': 0.3,  # Weight for metadata-based contrastive loss
    'spectral_augmentations': True,  # Enable spectral augmentations
    'temporal_augmentations': True,  # Enable temporal augmentations
    'noise_augmentations': True,  # Enable noise-based augmentations
    'save_checkpoints': True,
    'checkpoint_freq': 10,
    'num_workers': 4,  
    'pin_memory': True,  
    'gradient_accumulation_steps': 2, 
    'mixed_precision': True 
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
        mask_prob = self.augmentation_strength * mask_prob
        mask = np.random.random(len(spectrum)) > mask_prob
        masked_spectrum = spectrum * mask
        return masked_spectrum
    
    def intensity_scaling(self, spectrum, scale_range=None):
        """Apply intensity scaling (flux calibration variations)"""
        if scale_range is None:
            scale_range = 1.0 + self.augmentation_strength * 0.3
        scale_factor = np.random.uniform(1.0/scale_range, scale_range)
        return spectrum * scale_factor
    
    def create_augmented_view(self, spectrum):
        """Create an augmented view of the spectrum"""
        augmented = spectrum.copy()
        
        # Apply random combination of augmentations
        augmentations = []
        
        if self.config['noise_augmentations'] and np.random.random() > 0.5:
            augmented = self.add_noise(augmented)
            augmentations.append('noise')
        
        if self.config['spectral_augmentations'] and np.random.random() > 0.5:
            augmented = self.spectral_shift(augmented)
            augmentations.append('shift')
        
        if self.config['spectral_augmentations'] and np.random.random() > 0.5:
            augmented = self.spectral_stretch(augmented)
            augmentations.append('stretch')
        
        if self.config['spectral_augmentations'] and np.random.random() > 0.5:
            augmented = self.mask_regions(augmented)
            augmentations.append('mask')
        
        if self.config['spectral_augmentations'] and np.random.random() > 0.5:
            augmented = self.intensity_scaling(augmented)
            augmentations.append('scale')
        
        # Normalize the augmented spectrum
        if np.std(augmented) > 0:
            augmented = (augmented - np.mean(augmented)) / np.std(augmented)
        
        return augmented, augmentations

class ContrastiveDataset(Dataset):
    """Dataset for contrastive learning with augmented views"""
    
    def __init__(self, csv_path, spectra_dir, config, augmentations):
        self.csv_path = csv_path
        self.spectra_dir = spectra_dir
        self.config = config
        self.augmentations = augmentations
        
        # Load metadata
        self.df = pd.read_csv(csv_path)
        self.df['sobject_id'] = self.df['sobject_id'].astype(str)
        
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
            
            if not found:
                continue
        
        # Sample if needed
        if self.config['n_spectra'] and len(self.available_files) > self.config['n_spectra']:
            self.available_files = random.sample(self.available_files, self.config['n_spectra'])
        
        print(f"Available spectra for contrastive learning: {len(self.available_files)}")
        
        # Test a few spectra to ensure they load correctly
        print("Testing spectrum loading...")
        test_indices = self.available_files[:5]
        for idx in test_indices:
            try:
                test_spectrum = self.get_flux_from_all_cameras(self.df.iloc[idx]['sobject_id'])
                if test_spectrum is not None:
                    print(f"  ✓ Spectrum {self.df.iloc[idx]['sobject_id']}: {len(test_spectrum)} points")
                else:
                    print(f"  ✗ Spectrum {self.df.iloc[idx]['sobject_id']}: Failed to load")
            except Exception as e:
                print(f"  ✗ Spectrum {self.df.iloc[idx]['sobject_id']}: Error - {str(e)}")
        
        # Check if spectra directory exists and has files
        if not os.path.exists(self.spectra_dir):
            print(f"ERROR: Spectra directory does not exist: {self.spectra_dir}")
            raise FileNotFoundError(f"Spectra directory not found: {self.spectra_dir}")
        
        # List some FITS files to verify they exist
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
                        
                        # Ensure flux is valid
                        if flux is None or len(flux) == 0:
                            continue
                        
                        safe_flux = np.where(flux > 1e-10, flux, 1e-10)
                        flux_processed = np.log(safe_flux)
                        
                        # Check for invalid values
                        if np.any(np.isnan(flux_processed)) or np.any(np.isinf(flux_processed)):
                            continue
                        
                        all_wavelengths.append(wave)
                        all_fluxes.append(flux_processed)
                        cameras_found += 1
                        
                except Exception as e:
                    continue
            
            if len(all_wavelengths) == 0:
                return None, None
            
            # Concatenate all data
            full_wave = np.concatenate(all_wavelengths)
            full_flux = np.concatenate(all_fluxes)
            
            # Check for valid data
            if len(full_wave) == 0 or len(full_flux) == 0:
                return None, None
            
            sort_idx = np.argsort(full_wave)
            return full_wave[sort_idx], full_flux[sort_idx]
            
        except Exception as e:
            return None, None

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
                        
                        # Ensure flux is valid
                        if flux is None or len(flux) == 0:
                            continue
                        
                        safe_flux = np.where(flux > 1e-10, flux, 1e-10)
                        flux_processed = np.log(safe_flux)
                        
                        # Check for invalid values
                        if np.any(np.isnan(flux_processed)) or np.any(np.isinf(flux_processed)):
                            continue
                        
                        all_wavelengths.append(wave)
                        all_fluxes.append(flux_processed)
                except Exception as e:
                    continue
            
            if len(all_wavelengths) == 0:
                return None
            
            # Concatenate all data
            full_wave = np.concatenate(all_wavelengths)
            full_flux = np.concatenate(all_fluxes)
            
            # Check for valid data
            if len(full_wave) == 0 or len(full_flux) == 0:
                return None
            
            sort_idx = np.argsort(full_wave)
            full_wave = full_wave[sort_idx]
            full_flux = full_flux[sort_idx]
            
            # Interpolate to common grid
            try:
                interp_func = interp1d(full_wave, full_flux, kind='linear', 
                                      bounds_error=False, fill_value=0.0)
                interp_flux = interp_func(self.common_wavelength)
                
                # Check for invalid interpolated values
                if np.any(np.isnan(interp_flux)) or np.any(np.isinf(interp_flux)):
                    return None
                
                # Normalize
                if np.std(interp_flux) > 0:
                    interp_flux = (interp_flux - np.mean(interp_flux)) / np.std(interp_flux)
                
                # Final validation
                if len(interp_flux) != self.config['input_length']:
                    return None
                
                return interp_flux
                
            except Exception as e:
                return None
            
        except Exception as e:
            return None
    
    def __getitem__(self, idx):
        actual_idx = self.available_files[idx]
        row = self.df.iloc[actual_idx]
        sobject_id = row['sobject_id']
        
        # Load spectrum with retry logic
        max_retries = 3
        spectrum = None
        
        for attempt in range(max_retries):
            try:
                spectrum = self.get_flux_from_all_cameras(sobject_id)
                if spectrum is not None and len(spectrum) == self.config['input_length']:
                    break
                else:
                    spectrum = None
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Warning: Failed to load spectrum for {sobject_id} after {max_retries} attempts")
                    spectrum = None
                continue
        
        # Ensure we have a valid spectrum
        if spectrum is None or len(spectrum) != self.config['input_length']:
            # Create a safe fallback spectrum
            spectrum = np.random.normal(0, 1, self.config['input_length'])
            spectrum = (spectrum - np.mean(spectrum)) / np.std(spectrum)
        
        # Ensure spectrum has correct shape
        spectrum = np.array(spectrum, dtype=np.float32)
        if len(spectrum) != self.config['input_length']:
            # Pad or truncate if needed
            if len(spectrum) < self.config['input_length']:
                spectrum = np.pad(spectrum, (0, self.config['input_length'] - len(spectrum)), 'constant')
            else:
                spectrum = spectrum[:self.config['input_length']]
        
        # Create two augmented views
        view1, aug1 = self.augmentations.create_augmented_view(spectrum)
        view2, aug2 = self.augmentations.create_augmented_view(spectrum)
        
        # Ensure views have correct shape
        view1 = np.array(view1, dtype=np.float32)
        view2 = np.array(view2, dtype=np.float32)
        
        if len(view1) != self.config['input_length']:
            view1 = np.pad(view1, (0, self.config['input_length'] - len(view1)), 'constant')
        if len(view2) != self.config['input_length']:
            view2 = np.pad(view2, (0, self.config['input_length'] - len(view2)), 'constant')
        
        # Get metadata if available
        if self.config['use_metadata_contrast']:
            metadata = self.metadata_matrix[idx]
        else:
            metadata = np.zeros(len(self.metadata_columns))
        
        # Ensure metadata has correct shape
        metadata = np.array(metadata, dtype=np.float32)
        if len(metadata) != len(self.metadata_columns):
            metadata = np.zeros(len(self.metadata_columns), dtype=np.float32)
        
        # Ensure tensors have correct shape: (channels=1, length=input_length)
        view1_tensor = torch.FloatTensor(view1).unsqueeze(0)  # Add channel dimension
        view2_tensor = torch.FloatTensor(view2).unsqueeze(0)  # Add channel dimension
        
        return {
            'view1': view1_tensor,  # Shape: (1, 16384)
            'view2': view2_tensor,  # Shape: (1, 16384)
            'augmentations1': aug1,
            'augmentations2': aug2,
            'metadata': torch.FloatTensor(metadata),
            'sobject_id': sobject_id
        }

class ContrastiveEncoder(nn.Module):
    """Encoder based on VAE architecture for contrastive learning"""
    
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
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(self.flattened_size, config['projection_head_dim']),
            nn.ReLU(),
            nn.Linear(config['projection_head_dim'], config['latent_dim'])
        )
        
        # Metadata projection head (optional)
        if config['use_metadata_contrast']:
            self.metadata_projection = nn.Sequential(
                nn.Linear(5, 64),  # 5 stellar parameters
                nn.ReLU(),
                nn.Linear(64, config['latent_dim'])
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
    
    def encode_features(self, x):
        """Get features without projection (for downstream tasks)"""
        features = self.encoder(x)
        features = features.flatten(start_dim=1)
        return features

class ContrastiveLearningModel(nn.Module):
    """Complete contrastive learning model"""
    
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
    """Trainer for contrastive learning"""
    
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
            self.dataset = ContrastiveDataset(config['label_csv'], config['spectra_dir'], config, self.augmentations)
        # Custom collate function to handle shape inconsistencies
        def custom_collate(batch):
            # Ensure all tensors have consistent shapes
            batch_size = len(batch)
            
            # Initialize lists for each field
            view1_list = []
            view2_list = []
            metadata_list = []
            sobject_ids = []
            
            for item in batch:
                # Ensure view1 and view2 have correct shape: (channels=1, length=input_length)
                view1 = item['view1']  # Should already be (1, input_length)
                view2 = item['view2']  # Should already be (1, input_length)
                
                # Verify and fix shapes if needed
                if view1.dim() == 1:
                    view1 = view1.unsqueeze(0)  # Add channel dimension
                elif view1.dim() == 2 and view1.shape[0] != 1:
                    view1 = view1.unsqueeze(0)  # Ensure channel dimension is first
                
                if view2.dim() == 1:
                    view2 = view2.unsqueeze(0)  # Add channel dimension
                elif view2.dim() == 2 and view2.shape[0] != 1:
                    view2 = view2.unsqueeze(0)  # Ensure channel dimension is first
                
                # Ensure correct length
                if view1.shape[1] != config['input_length']:
                    if view1.shape[1] < config['input_length']:
                        padding = torch.zeros(1, config['input_length'] - view1.shape[1])
                        view1 = torch.cat([view1, padding], dim=1)
                    else:
                        view1 = view1[:, :config['input_length']]
                
                if view2.shape[1] != config['input_length']:
                    if view2.shape[1] < config['input_length']:
                        padding = torch.zeros(1, config['input_length'] - view2.shape[1])
                        view2 = torch.cat([view2, padding], dim=1)
                    else:
                        view2 = view2[:, :config['input_length']]
                
                view1_list.append(view1)
                view2_list.append(view2)
                metadata_list.append(item['metadata'])
                sobject_ids.append(item['sobject_id'])
            
            # Stack tensors - should result in (batch_size, channels=1, length=input_length)
            return {
                'view1': torch.stack(view1_list),  # Shape: (batch_size, 1, 16384)
                'view2': torch.stack(view2_list),  # Shape: (batch_size, 1, 16384)
                'metadata': torch.stack(metadata_list),
                'sobject_id': sobject_ids
            }
        
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=config['batch_size'], 
            shuffle=True, 
            num_workers=config['num_workers'],
            pin_memory=config['pin_memory'],
            collate_fn=custom_collate
        )
        
        # Initialize model
        self.model = ContrastiveLearningModel(config).to(self.device)
        self.criterion = ContrastiveLoss(config['temperature'], config['metadata_weight'])
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=config['learning_rate'], 
            weight_decay=1e-4  # L2 regularization to prevent overfitting
        )
        
        # Learning rate scheduler with warmup
        self.warmup_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer, 
            start_factor=0.1, 
            total_iters=10  # Warmup for first 10 epochs
        )
        
        # Main scheduler after warmup
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config['num_epochs'] - 10  # Start after warmup
        )
        
        # Mixed precision training for lab GPU efficiency
        if config['mixed_precision'] and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
            print("Mixed precision training enabled")
        else:
            self.scaler = None
        
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
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_spectral_loss = 0.0
        total_metadata_loss = 0.0
        
        for batch_idx, batch in enumerate(self.dataloader):
            # Move data to device
            view1 = batch['view1'].to(self.device)
            view2 = batch['view2'].to(self.device)
            metadata1 = batch['metadata'].to(self.device) if self.config['use_metadata_contrast'] else None
            metadata2 = batch['metadata'].to(self.device) if self.config['use_metadata_contrast'] else None
            
            # Debug tensor shapes (only for first few batches)
            if batch_idx < 3:
                print(f"  Debug - view1 shape: {view1.shape}, view2 shape: {view2.shape}")
                print(f"  Debug - view1 dtype: {view1.dtype}, view2 dtype: {view2.dtype}")
                print(f"  Debug - view1 range: [{view1.min():.3f}, {view1.max():.3f}]")
                print(f"  Debug - view2 range: [{view2.min():.3f}, {view2.max():.3f}]")
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(view1, view2, metadata1, metadata2)
                    loss_dict = self.criterion(
                        outputs['projections1'], 
                        outputs['projections2'],
                        outputs['metadata_projections1'],
                        outputs['metadata_projections2']
                    )
                
                # Backward pass with mixed precision
                self.optimizer.zero_grad()
                self.scaler.scale(loss_dict['total_loss']).backward()
                self.scaler.unscale_(self.optimizer)
                
                # Gradient clipping to prevent exploding gradients
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training without mixed precision
                outputs = self.model(view1, view2, metadata1, metadata2)
                loss_dict = self.criterion(
                    outputs['projections1'], 
                    outputs['projections2'],
                    outputs['metadata_projections1'],
                    outputs['metadata_projections2']
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss_dict['total_loss'].backward()
                
                # Gradient clipping to prevent exploding gradients
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss_dict['total_loss'].item()
            total_spectral_loss += loss_dict['spectral_loss'].item()
            total_metadata_loss += loss_dict['metadata_loss'].item()
            
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(self.dataloader)}: "
                      f"Loss={loss_dict['total_loss']:.4f}, "
                      f"Spectral={loss_dict['spectral_loss']:.4f}, "
                      f"Metadata={loss_dict['metadata_loss']:.4f}, "
                      f"Grad Norm={total_norm:.4f}")
        
        # Update learning rate with warmup
        if len(self.history['train_loss']) < 10:
            # Warmup phase
            self.warmup_scheduler.step()
        else:
            # Main scheduling phase
            self.scheduler.step()
        
        # Return average losses
        num_batches = len(self.dataloader)
        return {
            'total_loss': total_loss / num_batches,
            'spectral_loss': total_spectral_loss / num_batches,
            'metadata_loss': total_metadata_loss / num_batches,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def train(self):
        """Complete training loop"""
        print("Starting contrastive learning training...")
        print(f"Training for {self.config['num_epochs']} epochs")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Learning rate: {self.config['learning_rate']}")
        print(f"Temperature: {self.config['temperature']}")
        print(f"Metadata weight: {self.config['metadata_weight']}")
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            
            # Train
            epoch_losses = self.train_epoch()
            
            # Store history
            self.history['train_loss'].append(epoch_losses['total_loss'])
            self.history['spectral_loss'].append(epoch_losses['spectral_loss'])
            self.history['metadata_loss'].append(epoch_losses['metadata_loss'])
            self.history['learning_rate'].append(epoch_losses['learning_rate'])
            
            # Print epoch summary
            print(f"Epoch {epoch+1} Summary:")
            print(f"  Total Loss: {epoch_losses['total_loss']:.4f}")
            print(f"  Spectral Loss: {epoch_losses['spectral_loss']:.4f}")
            print(f"  Metadata Loss: {epoch_losses['metadata_loss']:.4f}")
            print(f"  Learning Rate: {epoch_losses['learning_rate']:.6f}")
            
            # Early stopping
            if epoch_losses['total_loss'] < best_loss - self.config['early_stopping_min_delta']:
                best_loss = epoch_losses['total_loss']
                patience_counter = 0
                
                # Save best model
                self.save_model('best_model.pth')
            else:
                patience_counter += 1
            
            # Save checkpoints
            if self.config['save_checkpoints'] and (epoch + 1) % self.config['checkpoint_freq'] == 0:
                self.save_model(f'checkpoint_epoch_{epoch+1}.pth')
            
            # Early stopping check
            if patience_counter >= self.config['early_stopping_patience']:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Save final model
        self.save_model('final_model.pth')
        
        # Save training history
        self.save_history()
        
        # Create training plots
        self.create_training_plots()
        
        print("Training completed!")
        
        # Return training results
        return {
            'best_train_loss': best_loss,
            'final_epoch': len(self.history['train_loss']),
            'early_stopped': patience_counter >= self.config['early_stopping_patience'],
            'train_loss_history': self.history['train_loss'],
            'spectral_loss_history': self.history['spectral_loss'],
            'metadata_loss_history': self.history['metadata_loss'],
            'learning_rate_history': self.history['learning_rate']
        }
    
    def save_model(self, filename):
        """Save model checkpoint"""
        os.makedirs(self.config['output_dir'], exist_ok=True)
        filepath = os.path.join(self.config['output_dir'], filename)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'history': self.history,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def save_history(self):
        """Save training history"""
        os.makedirs(self.config['output_dir'], exist_ok=True)
        history_file = os.path.join(self.config['output_dir'], 'training_history.json')
        
        # Convert numpy arrays to lists for JSON serialization
        history_dict = {}
        for key, value in self.history.items():
            if isinstance(value, list):
                history_dict[key] = [float(v) for v in value]
        
        with open(history_file, 'w') as f:
            json.dump(history_dict, f, indent=4)
        
        print(f"Training history saved to {history_file}")
    
    def create_training_plots(self):
        """Create training visualization plots"""
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Contrastive Learning Training Progress', fontsize=16)
        
        # Loss curves
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Total loss
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', linewidth=2, label='Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Total Training Loss')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Component losses
        axes[0, 1].plot(epochs, self.history['spectral_loss'], 'r-', linewidth=2, label='Spectral Loss')
        axes[0, 1].plot(epochs, self.history['metadata_loss'], 'g-', linewidth=2, label='Metadata Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Component Losses')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Learning rate
        axes[1, 0].plot(epochs, self.history['learning_rate'], 'm-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Loss comparison
        axes[1, 1].plot(epochs, self.history['train_loss'], 'b-', linewidth=2, label='Training Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Training Progress')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plot_file = os.path.join(self.config['output_dir'], 'training_progress.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training plots saved to {plot_file}")

def train_single_contrastive_model(latent_dim, base_config):
    """Train a single contrastive learning model with specific latent dimension"""
    print(f"\n{'='*60}")
    print(f"TRAINING CONTRASTIVE MODEL - LATENT DIMENSION: {latent_dim}")
    print(f"{'='*60}")
    
    # Create experiment-specific config
    experiment_config = base_config.copy()
    experiment_config['latent_dim'] = latent_dim
    experiment_config['output_dir'] = os.path.join(base_config['output_dir'], f'L_{latent_dim}')
    
    # Create output directory
    os.makedirs(experiment_config['output_dir'], exist_ok=True)
    
    try:
        # Create dataset first to get dynamic input_length
        print("Creating dataset to determine dynamic input length...")
        temp_dataset = ContrastiveDataset(
            experiment_config['label_csv'], 
            experiment_config['spectra_dir'], 
            experiment_config, 
            SpectralAugmentations(experiment_config)
        )
        
        # Update config with dynamic input_length
        experiment_config['input_length'] = temp_dataset.config['input_length']
        print(f"Dynamic input length determined: {experiment_config['input_length']}")
        
        # Create trainer with specific latent dimension and updated config
        # Pass the already-created dataset to avoid recreating it
        trainer = ContrastiveTrainer(experiment_config, latent_dim=latent_dim, dataset=temp_dataset)
        
        # Start training
        results = trainer.train()
        
        # Save training results
        results['config'] = experiment_config
        results['latent_dim'] = latent_dim
        
        # Save results to file
        results_path = os.path.join(experiment_config['output_dir'], 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        
        print(f"\nTraining completed for latent dimension {latent_dim}")
        print(f"Results saved to: {experiment_config['output_dir']}")
        
        return results
        
    except Exception as e:
        print(f"Error training model with latent dimension {latent_dim}: {e}")
        return None

def main():
    """Main function to run contrastive learning with multiple latent dimensions"""
    print("=" * 60)
    print("STELLAR SPECTRA CONTRASTIVE LEARNING - MULTIPLE LATENT DIMS")
    print("=" * 60)
    
    # Set random seed
    random.seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])
    torch.manual_seed(CONFIG['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(CONFIG['random_seed'])
        torch.cuda.manual_seed_all(CONFIG['random_seed'])
    
    # Train multiple models with different latent dimensions
    all_results = {}
    
    for latent_dim in CONFIG['latent_dims']:
        try:
            results = train_single_contrastive_model(latent_dim, CONFIG)
            if results:
                all_results[latent_dim] = results
        except Exception as e:
            print(f"Error training model with latent dimension {latent_dim}: {e}")
            continue
    
    # Save summary of all experiments
    if all_results:
        summary_path = os.path.join(CONFIG['output_dir'], 'experiment_summary.json')
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        
        summary = {
            'experiment_date': str(datetime.now()),
            'models_trained': len(all_results),
            'latent_dimensions': list(all_results.keys()),
            'results_summary': {
                str(latent_dim): {
                    'best_train_loss': results.get('best_train_loss', 'N/A'),
                    'final_epoch': results.get('final_epoch', 'N/A'),
                    'early_stopped': results.get('early_stopped', False)
                }
                for latent_dim, results in all_results.items()
            }
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        # Create comparison plot
        plt.figure(figsize=(12, 8))
        for latent_dim, results in all_results.items():
            if 'train_loss_history' in results:
                plt.plot(results['train_loss_history'], label=f'L={latent_dim}', marker='o', markersize=3)
        
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.title('Comparison of Training Loss Across Latent Dimensions')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(CONFIG['output_dir'], 'comparison_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n{'='*60}")
        print("ALL CONTRASTIVE LEARNING EXPERIMENTS COMPLETED!")
        print(f"{'='*60}")
        print(f"Models trained: {len(all_results)}")
        print(f"Latent dimensions: {list(all_results.keys())}")
        print(f"Summary saved to: {summary_path}")
        print(f"Comparison plot saved to: {os.path.join(CONFIG['output_dir'], 'comparison_plot.png')}")
    else:
        print("No models were successfully trained.")

if __name__ == "__main__":
    main() 