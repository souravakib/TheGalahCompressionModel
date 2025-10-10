# GalahCompressionModel

A comprehensive deep learning framework for stellar spectral compression and analysis using multiple neural network architectures. This project implements and compares different approaches to compress and learn representations from stellar spectra from the GALAH (Galactic Archaeology with HERMES) survey.

## Overview

This repository contains implementations of three main approaches for stellar spectral compression:

1. **CNN Autoencoder** - Traditional convolutional autoencoder for spectral compression
2. **Variational Autoencoder (VAE)** - Probabilistic autoencoder with latent space modeling
3. **Contrastive Learning** - Self-supervised representation learning for stellar spectra

## Features

- **Multi-Architecture Support**: Compare different neural network approaches for spectral compression
- **Flexible Latent Dimensions**: Support for various latent space sizes (16, 32, 64, 128, 256, 512, 1024)
- **GALAH Survey Integration**: Direct integration with GALAH FITS files and metadata
- **Comprehensive Evaluation**: Built-in metrics for compression ratio, reconstruction quality, and downstream task performance
- **GPU/CPU Support**: Automatic device detection and optimization
- **Reproducible Results**: Seeded random number generation for consistent experiments

## File Structure

```
├── galah_cnn_autoencoder.py              # CNN Autoencoder implementation
├── galah_variational_autoencoder.py      # Original VAE implementation
├── galah_variational_autoencoder_improved.py  # Improved VAE with better training dynamics
├── galah_contrastive_learning.py         # Contrastive learning implementation
├── galah_contrastive_learning_improved.py # Improved contrastive learning with fixes
└── README.md                             # This file
```

## Models

### 1. CNN Autoencoder (`galah_cnn_autoencoder.py`)

Traditional convolutional autoencoder for stellar spectral compression.

**Key Features:**
- 3-layer CNN encoder/decoder architecture
- Configurable latent dimensions: [16, 32, 64, 128, 256]
- Early stopping and learning rate scheduling
- Masked reconstruction loss for handling missing spectral regions

**Architecture:**
- Encoder: 1D Convolutions with BatchNorm and ReLU
- Decoder: 1D Transpose Convolutions with BatchNorm and ReLU
- Latent space: Fully connected layer

### 2. Variational Autoencoder (`galah_variational_autoencoder.py`)

Probabilistic autoencoder that learns a latent distribution for stellar spectra.

**Key Features:**
- KL divergence regularization with annealing
- Free bits implementation for stable training
- Dynamic wavelength grid creation
- Masked reconstruction loss
- Support for latent dimensions: [16, 32, 64, 128, 256]

**Training Features:**
- KL annealing over 50 epochs
- Free bits threshold (0.25)
- Early stopping with patience

### 3. Improved VAE (`galah_variational_autoencoder_improved.py`)

Enhanced version of the VAE with better training dynamics and stability.

**Improvements:**
- Fixed input length (16384 points) for consistency
- Simplified architecture for better convergence
- Disabled early stopping for full training
- Optimized for latent dimensions: [64, 512, 1024]
- Linux lab environment configuration

### 4. Contrastive Learning (`galah_contrastive_learning.py`)

Self-supervised representation learning using contrastive objectives.

**Key Features:**
- SimCLR-style contrastive learning
- Spectral and temporal augmentations
- Metadata-based contrastive learning
- Temperature-scaled similarity learning
- Support for latent dimensions: [16, 32, 64, 128, 256, 512, 1024]

**Augmentations:**
- Spectral noise injection
- Wavelength shifting
- Flux scaling
- Temporal masking

### 5. Improved Contrastive Learning (`galah_contrastive_learning_improved.py`)

Enhanced contrastive learning with fixes for higher dimensional latent spaces.

**Improvements:**
- Better handling of higher latent dimensions (256, 512, 1024)
- Improved projection head architecture
- Enhanced augmentation strategies
- More stable training dynamics

## Data Format

The models expect GALAH survey data in the following format:

- **FITS Files**: Stellar spectra in FITS format with flux data in extension 4
- **Metadata CSV**: Stellar parameters (Teff, logg, Fe/H) and object IDs
- **File Naming**: `{sobject_id}{camera}.fits` where camera ∈ [1,2,3,4]

## Usage

### Basic Training

```python
# CNN Autoencoder
python galah_cnn_autoencoder.py

# Variational Autoencoder
python galah_variational_autoencoder.py

# Improved VAE
python galah_variational_autoencoder_improved.py

# Contrastive Learning
python galah_contrastive_learning.py

# Improved Contrastive Learning
python galah_contrastive_learning_improved.py
```

### Configuration

Each model includes a `CONFIG` dictionary at the top of the file with customizable parameters:

```python
CONFIG = {
    'latent_dims': [16, 32, 64, 128, 256],  # Latent dimensions to train
    'channels': '32,64,128',               # CNN channels
    'kernels': '128,64,32',                # CNN kernel sizes
    'pools_or_strides': '8,8,8',           # Downsampling strides
    'learning_rate': 1e-5,                 # Learning rate
    'batch_size': 32,                      # Batch size
    'num_epochs': 150,                     # Training epochs
    'n_spectra': 4000,                     # Number of spectra to use
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    # ... other parameters
}
```

## Requirements

```bash
torch>=1.9.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
astropy>=4.0.0
scipy>=1.7.0
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd GalahCompressionModel
```

2. Install dependencies:
```bash
pip install torch numpy pandas matplotlib scikit-learn astropy scipy
```

3. Update file paths in the configuration:
```python
CONFIG = {
    'LABEL_CSV': 'path/to/your/metadata.csv',
    'SPECTRA_DIR': 'path/to/your/fits/files',
    # ... other paths
}
```

## Results

Each model generates comprehensive results including:

- **Training History**: Loss curves and convergence metrics
- **Reconstructions**: Original vs reconstructed spectra visualizations
- **Latent Space**: 2D projections of learned representations
- **Architecture Info**: Model parameters and configuration
- **Performance Metrics**: Compression ratios and reconstruction quality

### Experimental Results

All experimental results are available on Google Drive:
**[📁 Download Results](https://drive.google.com/drive/folders/1ooxaoOCD0piFRq9bxgNmpBYROHINz_mD?usp=drive_link)**

The results are organized into three main folders:

1. **Contrastive Learning** - Results from contrastive learning experiments
2. **unsupervised_cnn_ae** - CNN autoencoder compression results  
3. **VAE** - Variational autoencoder results and analysis

Results are saved in organized directories:
```
results/
├── unsupervised_cnn_ae/
├── variational_autoencoder/
├── variational_autoencoder_v7/
├── contrastive_learning/
└── contrastive_learning_improved/
```

## Model Comparison

The repository enables systematic comparison of different approaches:

| Model | Latent Dims | Compression Ratio | Reconstruction Quality | Training Stability |
|-------|-------------|-------------------|----------------------|-------------------|
| CNN AE | 16-256 | High | Good | Stable |
| VAE | 16-256 | High | Good | Moderate |
| VAE Improved | 64-1024 | Very High | Very Good | Stable |
| Contrastive | 16-1024 | Variable | Excellent | Stable |
| Contrastive Improved | 16-1024 | Variable | Excellent | Very Stable |

### Experimental Findings

Based on the comprehensive experiments conducted (available in the [Google Drive results](https://drive.google.com/drive/folders/1ooxaoOCD0piFRq9bxgNmpBYROHINz_mD?usp=drive_link)):

- **CNN Autoencoder**: Achieves good compression ratios with stable training across all latent dimensions
- **VAE Models**: Show improved reconstruction quality with probabilistic latent representations, with the improved version handling higher dimensions more effectively
- **Contrastive Learning**: Demonstrates excellent representation learning capabilities, particularly useful for downstream tasks like stellar parameter estimation
- **Higher Latent Dimensions**: The improved models (VAE v7 and Contrastive Learning Improved) successfully handle dimensions up to 1024, providing very high compression ratios while maintaining quality

## Research Applications

This framework is designed for:

- **Stellar Parameter Estimation**: Using compressed representations for Teff, logg, Fe/H prediction
- **Stellar Classification**: Binary/multi-class classification tasks
- **Survey Data Compression**: Efficient storage and transmission of large spectral datasets
- **Anomaly Detection**: Identifying unusual stellar spectra
- **Representation Learning**: Learning meaningful stellar spectral features

## Citation

If you use this code in your research, please cite:

```bibtex
@software{galah_compression_model,
  title={GalahCompressionModel: Deep Learning Framework for Stellar Spectral Compression},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[your-username]/GalahCompressionModel}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- GALAH Survey team for providing the stellar spectral data
- PyTorch team for the deep learning framework
- Astropy project for astronomical data handling tools

## Contact

For questions or collaboration, please contact s.souravakib@gmail.com
