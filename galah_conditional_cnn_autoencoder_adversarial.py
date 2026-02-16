"""
Conditional CNN Autoencoder with Adversarial Training
Uses adversarial discriminator to penalize teff/logg information in latent space
"""

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

# Import from the original file to avoid duplication
from galah_conditional_cnn_autoencoder import (
    set_seed,
    get_flux_from_all_cameras,
    ConditionalSpectraDataset,
    LABEL_CSV,
    SPECTRA_DIR
)

# --- Configuration ---
config = {
    "experiment_name": "conditional_cnn_ae_adversarial_l_16",
    "downsampling_mode": "stride",
    "pools_or_strides": "8,8,8", 
    "kernels": "128,64,32",
    "channels": "32,64,128",
    "latent_dim": 16,
    "batch_size": 32,
    "epochs": 50,
    "lr": 1e-4,
    "early_stopping_patience": 10,
    "early_stopping_min_delta": 0.0001,
    "n_spectra": 2000,
    "output_dir": "./results/conditional_cnn_ae_adversarial/L_16",
    "random_seed": 42,
    "adversarial_weight": 0.1,  # Weight for adversarial loss (penalize teff/logg in latent)
    "grl_lambda": 1.0,  # Gradient reversal strength for nuisance head
    "use_adversarial": True  # Enable/disable adversarial training
}

# List of latent dimensions to experiment with
LATENT_DIMS = [16]
# Adversarial weights to sweep
ADVERSARIAL_WEIGHTS = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

# Conditional CNN Autoencoder with Adversarial Discriminator
class ConditionalCNNAutoencoderAdversarial(nn.Module):
    def __init__(self, config, debug_shapes=False):
        super().__init__()
        channels = [int(c) for c in config['channels'].split(',')]
        kernels = [int(k) for k in config['kernels'].split(',')]
        strides = [int(s) for s in config['pools_or_strides'].split(',')]
        num_layers = len(channels)
        input_length = config['input_length']
        
        self.debug_shapes = debug_shapes
        self.input_length = input_length
        
        self.stellar_param_dim = 2  # T_eff and log_g
        
        # Encoder: all conv blocks on spectrum only (no early conditioning)
        self.encoder_layers = nn.ModuleList()
        in_channels = 1
        current_length = input_length
        encoder_output_lengths = []

        if debug_shapes:
            print(f"Encoder size tracking:")
            print(f"  Input length: {input_length}")
            print(f"  Stellar param dimension: {self.stellar_param_dim}")

        for i, (c, k, s) in enumerate(zip(channels, kernels, strides)):
            padding = (k - 1) // 2
            self.encoder_layers.append(nn.Conv1d(in_channels, c, kernel_size=k, stride=s, padding=padding))
            self.encoder_layers.append(nn.BatchNorm1d(c))
            self.encoder_layers.append(nn.ReLU())

            current_length = (current_length + (2 * padding) - k) // s + 1
            encoder_output_lengths.append(current_length)

            if debug_shapes:
                print(f"  Layer {i}: {current_length} (kernel={k}, stride={s}, padding={padding})")

            in_channels = c

        self.enc_out_channels = channels[-1]
        self.enc_out_length = encoder_output_lengths[-1]
        flattened_size = self.enc_out_channels * self.enc_out_length
        
        # Bottleneck
        self.to_latent = nn.Linear(flattened_size, config['latent_dim'])
        self.from_latent = nn.Linear(config['latent_dim'] + self.stellar_param_dim, flattened_size)
        
        # Gradient reversal + nuisance head (predict teff/logg from latent)
        self.grl = GradientReversalLayer(lambda_=config.get('grl_lambda', 1.0))
        self.nuisance_head = nn.Sequential(
            nn.Linear(config['latent_dim'], 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.stellar_param_dim),  # Predict teff and logg
        )
        
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
            padding = (k - 1) // 2
            
            input_len = rev_encoder_output_lengths[i]
            output_padding = target_length - (((input_len - 1) * s) + k - (2 * padding))
            
            if debug_shapes:
                print(f"  Decoder layer {i}: input_len={input_len}, target={target_length}, output_padding={output_padding}")
            
            if output_padding < 0:
                output_padding = 0
                print(f"Warning: Adjusted output_padding to 0 for decoder layer {i}")
            elif output_padding > 1:
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
    
    def forward(self, x, abundances, stellar_params):
        """Forward pass with decoder conditioning and adversarial nuisance head."""
        if self.debug_shapes and hasattr(self, '_debug_printed'):
            return self._forward_without_debug(x, abundances, stellar_params)
        
        if self.debug_shapes:
            print(f"Input shape: {x.shape}")
            print(f"Abundances shape: {abundances.shape}")
            print(f"Stellar params shape: {stellar_params.shape}")
        
        # Encode spectrum through all conv blocks
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)
            if self.debug_shapes:
                print(f"Encoder Layer {i} ({layer.__class__.__name__}): {x.shape}")
        
        x = x.flatten(1)
        if self.debug_shapes:
            print(f"Flattened shape: {x.shape}")
        
        # Bottleneck
        latent = self.to_latent(x)
        if self.debug_shapes:
            print(f"Latent vector shape: {latent.shape}")

        nuisance_pred = self.nuisance_head(self.grl(latent))
        if self.debug_shapes:
            print(f"Nuisance pred shape: {nuisance_pred.shape}")

        latent_cond = torch.cat([latent, stellar_params], dim=1)
        x = self.from_latent(latent_cond)
        x = x.view(-1, self.enc_out_channels, self.enc_out_length)
        if self.debug_shapes:
            print(f"From latent reshaped: {x.shape}")
        
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x)
            if self.debug_shapes:
                print(f"Decoder Layer {i} ({layer.__class__.__name__}): {x.shape}")
        
        if self.debug_shapes:
            self._debug_printed = True
        
        return x.squeeze(1), latent, nuisance_pred
    
    def _forward_without_debug(self, x, abundances, stellar_params):
        """Forward pass without debug printing for training"""
        for layer in self.encoder_layers:
            x = layer(x)
        x = x.flatten(1)
        latent = self.to_latent(x)
        
        nuisance_pred = self.nuisance_head(self.grl(latent))

        latent_cond = torch.cat([latent, stellar_params], dim=1)
        x = self.from_latent(latent_cond)
        x = x.view(-1, self.enc_out_channels, self.enc_out_length)
        
        for layer in self.decoder_layers:
            x = layer(x)
        
        return x.squeeze(1), latent, nuisance_pred

# --- Main training function ---
def main():
    set_seed(config['random_seed'])
    
    all_results = {}
    
    for latent_dim in LATENT_DIMS:
        for adv_weight in ADVERSARIAL_WEIGHTS:
            try:
                results = train_single_model(latent_dim, adv_weight, config)
                if results:
                    all_results[(latent_dim, adv_weight)] = results
            except Exception as e:
                print(f"Error training model with latent dimension {latent_dim} and adversarial weight {adv_weight}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Models trained: {len(all_results)}")
    print(f"(latent_dim, adversarial_weight) runs: {list(all_results.keys())}")

def train_single_model(latent_dim, adversarial_weight, base_config):
    """Train a single model with given latent dimension and save results"""
    
    experiment_config = base_config.copy()
    experiment_config['latent_dim'] = latent_dim
    experiment_config['adversarial_weight'] = adversarial_weight
    experiment_config['experiment_name'] = f"conditional_cnn_ae_adversarial_l_{latent_dim}_w_{adversarial_weight}"
    experiment_config['output_dir'] = f"./results/conditional_cnn_ae_adversarial/L_{latent_dim}/w_{adversarial_weight}"
    
    print(f"\n{'='*60}")
    print(f"Training Conditional CNN Autoencoder (Adversarial) with latent dimension: {latent_dim}")
    print(f"Output directory: {experiment_config['output_dir']}")
    print(f"Adversarial weight: {experiment_config.get('adversarial_weight', 0.1)}")
    print(f"{'='*60}")
    
    os.makedirs(experiment_config['output_dir'], exist_ok=True)
    
    if not os.path.exists(SPECTRA_DIR):
        print(f"Error: Spectra directory not found: {SPECTRA_DIR}")
        return None
    
    # Load and prepare data (same as original)
    df = pd.read_csv(LABEL_CSV)
    df['sobject_id'] = df['sobject_id'].astype(str)
    
    abundance_cols = [
        'Fe_h', 'Mg_fe', 'Si_fe', 'Ca_fe', 'Ti_fe',
        'Al_fe', 'K_fe', 'Mn_fe', 'Zn_fe', 'Ba_fe', 'Eu_fe'
    ]
    abundance_cols = [c for c in abundance_cols if c in df.columns]
    
    required_cols = ['teff', 'logg'] + abundance_cols
    initial_count = len(df)
    df = df.dropna(subset=required_cols)
    filtered_count = len(df)
    if filtered_count < initial_count:
        print(f"Filtered out {initial_count - filtered_count} rows with missing required columns")
        print(f"Remaining samples: {filtered_count}")
    
    if experiment_config['n_spectra'] is not None:
        df = df.head(experiment_config['n_spectra'])
    
    # Prepare common grid
    sample_ids = df['sobject_id'].sample(n=min(50, len(df)), random_state=experiment_config['random_seed']).tolist()
    sample_waves = [wave for sid in sample_ids if (wave := get_flux_from_all_cameras(sid, SPECTRA_DIR)[0]) is not None]
    common_grid = sample_waves[np.argsort([len(w) for w in sample_waves])[len(sample_waves)//2]]
    experiment_config['input_length'] = len(common_grid)
    experiment_config['abundance_dim'] = len(abundance_cols)
    print(f"Using common grid length: {experiment_config['input_length']}")
    print(f"Number of abundance columns: {len(abundance_cols)}")
    
    # Split data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=experiment_config['random_seed'])
    
    # Compute normalization parameters
    train_teff_mean = train_df['teff'].mean()
    train_teff_std = train_df['teff'].std()
    train_logg_mean = train_df['logg'].mean()
    train_logg_std = train_df['logg'].std()
    
    train_abundance_means = {}
    train_abundance_stds = {}
    for col in abundance_cols:
        train_abundance_means[col] = train_df[col].mean()
        train_abundance_stds[col] = train_df[col].std()
    
    print(f"T_eff normalization: mean={train_teff_mean:.2f}, std={train_teff_std:.2f}")
    print(f"log_g normalization: mean={train_logg_mean:.2f}, std={train_logg_std:.2f}")
    print(f"Abundance normalization computed for {len(abundance_cols)} columns")
    
    # Create datasets
    train_dataset = ConditionalSpectraDataset(
        train_df, SPECTRA_DIR, common_grid,
        teff_mean=train_teff_mean, teff_std=train_teff_std,
        logg_mean=train_logg_mean, logg_std=train_logg_std,
        abundance_cols=abundance_cols,
        abundance_means=train_abundance_means,
        abundance_stds=train_abundance_stds
    )
    val_dataset = ConditionalSpectraDataset(
        val_df, SPECTRA_DIR, common_grid,
        teff_mean=train_teff_mean, teff_std=train_teff_std,
        logg_mean=train_logg_mean, logg_std=train_logg_std,
        abundance_cols=abundance_cols,
        abundance_means=train_abundance_means,
        abundance_stds=train_abundance_stds
    )
    
    train_loader = DataLoader(train_dataset, batch_size=experiment_config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=experiment_config['batch_size'], shuffle=False)
    
    # Setup model and training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConditionalCNNAutoencoderAdversarial(experiment_config, debug_shapes=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=experiment_config['lr'])
    
    # Training variables
    best_val_loss = float('inf')
    patience_counter = 0
    loss_history, val_loss_history = [], []
    
    # Training loop
    for epoch in range(experiment_config['epochs']):
        model.train()
        running_loss = 0.0
        running_recon_loss = 0.0
        running_adv_loss = 0.0
        size_mismatch_reported = False
        
        for batch, mask, abundances, stellar_params in train_loader:
            batch = batch.to(device)
            mask = mask.to(device)
            abundances = abundances.to(device)
            stellar_params = stellar_params.to(device)
            optimizer.zero_grad()
            output, latent, nuisance_pred = model(batch.unsqueeze(1), abundances, stellar_params)
            
            # Handle size mismatch
            min_size = min(output.shape[1], batch.shape[1])
            if output.shape[1] != batch.shape[1] and not size_mismatch_reported:
                print(f"Epoch {epoch+1}: Size mismatch: output={output.shape[1]}, batch={batch.shape[1]}, using first {min_size} elements")
                size_mismatch_reported = True
            
            # Loss calculation
            output_trimmed = output[:, :min_size]
            batch_trimmed = batch[:, :min_size]
            mask_trimmed = mask[:, :min_size]
            
            reconstruction_loss = (((output_trimmed - batch_trimmed) ** 2) * mask_trimmed).sum() / mask_trimmed.sum()
            
            # Adversarial loss: penalize latent from encoding teff/logg
            total_loss = reconstruction_loss
            adversarial_loss_val = 0.0
            
            if experiment_config.get('use_adversarial', False):
                adversarial_weight = experiment_config.get('adversarial_weight', 0.1)
                nuisance_loss = nn.MSELoss()(nuisance_pred, stellar_params)
                total_loss = reconstruction_loss + adversarial_weight * nuisance_loss
                adversarial_loss_val = nuisance_loss.item()
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += reconstruction_loss.item()
            running_recon_loss += reconstruction_loss.item()
            running_adv_loss += adversarial_loss_val
        
        train_loss = running_loss / len(train_loader)
        train_recon_loss = running_recon_loss / len(train_loader)
        train_adv_loss = running_adv_loss / len(train_loader)
        loss_history.append(train_loss)
        
        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch, mask, abundances, stellar_params in val_loader:
                batch = batch.to(device)
                mask = mask.to(device)
                abundances = abundances.to(device)
                stellar_params = stellar_params.to(device)
                output, _, _ = model(batch.unsqueeze(1), abundances, stellar_params)
                
                min_size = min(output.shape[1], batch.shape[1])
                output_trimmed = output[:, :min_size]
                batch_trimmed = batch[:, :min_size]
                mask_trimmed = mask[:, :min_size]
                
                val_loss = (((output_trimmed - batch_trimmed) ** 2) * mask_trimmed).sum() / mask_trimmed.sum()
                running_val_loss += val_loss.item()
        
        val_loss = running_val_loss / len(val_loader)
        val_loss_history.append(val_loss)
        
        loss_info = f"Epoch {epoch+1}/{experiment_config['epochs']} — Train Loss: {train_loss:.6f} (Recon: {train_recon_loss:.6f}"
        if experiment_config.get('use_adversarial', False):
            loss_info += f", Adv: {train_adv_loss:.6f}"
        loss_info += f") — Val Loss: {val_loss:.6f}"
        print(loss_info)
        
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
        'early_stopped': patience_counter >= experiment_config['early_stopping_patience'],
        'normalization_params': {
            'teff_mean': float(train_teff_mean),
            'teff_std': float(train_teff_std),
            'logg_mean': float(train_logg_mean),
            'logg_std': float(train_logg_std),
            'abundance_cols': abundance_cols,
            'abundance_means': {k: float(v) for k, v in train_abundance_means.items()},
            'abundance_stds': {k: float(v) for k, v in train_abundance_stds.items()}
        }
    }
    
    with open(os.path.join(experiment_config['output_dir'], 'training_logs.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Save loss curve plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_history) + 1), loss_history, label='Train Loss', marker='o')
    plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label='Val Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training & Validation Loss - Conditional AE (Adversarial, Latent Dim: {latent_dim})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(experiment_config['output_dir'], 'loss_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training completed for latent dimension {latent_dim}")
    print(f"Results saved to: {experiment_config['output_dir']}")
    
    return results

if __name__ == '__main__':
    main()

