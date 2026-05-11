# CTGAN Shifter

A neural network module for **meta-feature-conditioned latent space steering** of a pretrained CTGAN — generating tabular data with user-specified statistical properties.

This module was originally developed in [latent-tabular-lens](https://github.com/Naramerk/latent-tabular-lens) and has been integrated into this unified OOD evaluation repository.

## Overview

Standard CTGAN samples noise `Z ~ N(0, I)` and passes it through a frozen generator. **Shifter** learns to apply a small, targeted shift to `Z` so that the generated data matches a desired set of meta-features `m*`. The CTGAN weights are **frozen throughout** — only Shifter is trained.

## Architecture

- Residual shift: `z̃_i = z_i + δ_scale · Δ_θ([z_i, c, μ_Z])` where `c = MetaEncoder(m*)`
- Loss: `L = MSE(m̂, m*) + λ_Z · |Z̃ - Z|² + λ_X · |X̃ - X_base|²`

## Directory Structure

```
ctgan_shifter/
├── src/
│   ├── shifter.py          # Main Shifter neural network
│   ├── ctgan_adapter.py    # CTGAN adapter for differentiable generation
│   └── differentiable_mfe.py  # Differentiable meta-feature extraction
├── preprocessing/
│   └── tab_preprocessing.py   # Tabular data preprocessing utilities
├── external/
│   └── ctgan_repo/         # External CTGAN library dependency
│       ├── ctgan/          # CTGAN package
│       ├── latest_requirements.txt
│       └── pyproject.toml
├── example/
│   ├── shifter_electricity_demo.ipynb  # Demo notebook (Electricity dataset)
│   ├── synthetic_shifted.csv           # Example output: shifted synthetic data
│   ├── shifter.pt                      # Pretrained Shifter weights
│   └── trained_ctgan_iris.pkl          # Pretrained CTGAN (Iris dataset)
└── README.md
```

## Key Components

### `src/shifter.py`
The core Shifter network. Takes latent noise vectors and target meta-features as input, outputs shifted latent vectors that produce data with the desired statistical properties.

### `src/ctgan_adapter.py`
Adapter class for interfacing with the CTGAN generator in a differentiable way, allowing gradient flow through the generation process during Shifter training.

### `src/differentiable_mfe.py`
Differentiable implementation of meta-feature extraction (mean, entropy, etc.) using PyTorch operations, enabling end-to-end training.

### `preprocessing/tab_preprocessing.py`
Utilities for preprocessing tabular data: type checking, handling NaN values, encoding categorical columns.

## Usage

See `example/shifter_electricity_demo.ipynb` for a complete demo using the Electricity dataset.

### Quick Start

```python
from src.ctgan_adapter import CTGANRepoAdapter
from src.shifter import Shifter
from src.differentiable_mfe import extract_torch_mfe

# Load pretrained CTGAN
ctgan = torch.load('example/trained_ctgan_iris.pkl', map_location=device)
adapter = CTGANRepoAdapter(ctgan)

# Train Shifter to match target meta-features
shifter = Shifter(z_dim=adapter.z_dim, m_dim=len(target_meta), ...)
# ... training loop ...

# Generate shifted synthetic data
with torch.no_grad():
    Z = torch.randn(1, n_rows, shifter.z_dim)
    Z_tilde = shifter(Z, target_meta.unsqueeze(0))
    df_synth = adapter.generate_from_noise(Z_tilde.reshape(-1, shifter.z_dim).numpy())
```

## Dependencies

- PyTorch
- scikit-learn
- pandas
- numpy
- CTGAN (included in `external/ctgan_repo/`)
