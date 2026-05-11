# Evaluating robustness of tabular models under meta-features based shifts

## 🔍 Overview

This repository implements a universal, reproducible Out-of-Distribution (OOD) evaluation protocol for tabular data. It uses evolutionary optimization to create train-test splits that maximize meta-feature differences, enabling controlled investigation of model behavior under distributional shifts.

This is the unified thesis repository, consolidating the OOD evaluation framework and the CTGAN-based distribution shifter (`ctgan_shifter`) originally developed in [latent-tabular-lens](https://github.com/Naramerk/latent-tabular-lens).

## ✨ Key Features

- **📊 Meta-Feature Based Splitting**: Optimizes data splits using dataset characteristics like mutual information and class concentration
- **🧬 Synthetic Data Generation**: Creates synthetic datasets matching target meta-feature distributions
- **🛡️ Robust Model Evaluation**: Includes IRM and DRO model implementations for OOD testing
- **📈 Comprehensive Benchmarking**: Tests on real-world tabular datasets with known shifts
- **🔀 CTGAN-Based Distribution Shifter**: Latent-space distribution shifting using CTGAN for controlled OOD sample generation

## 🎯 Problem Statement

In empirical machine learning settings, the core assumption that training and test distributions are identical is often violated. This is particularly challenging in high-stakes domains (medical diagnostics, finance, climate monitoring) where model performance degradation under distributional shifts can have significant real-world implications.

Traditional tabular datasets lack mechanisms for constructing well-defined distributional shifts, making systematic OOD evaluation difficult. This work addresses this gap by introducing a principled protocol that enables controlled manipulation of dataset characteristics.

## 🧠 Methodology

### Proposed Approach

Our approach enhances OOD evaluation through meta-feature based splitting, enabling controlled distributional shifts without architectural changes. Unlike random splits that may not capture meaningful distributional differences, our evolutionary algorithm systematically constructs train-test partitions that maximize meta-feature disparities. The method applies constraints through the fitness function rather than modifying the data generation process, maintaining dataset integrity while enforcing interpretable geometric relationships. Additionally, our framework supports synthetic data generation that preserves specific meta-feature distributions, allowing researchers to create controlled datasets with desired statistical properties for comprehensive robustness testing.

## 1️⃣ Meta-Feature Based Splitting

The core idea is formulating train-test partitioning as an optimization problem:

```
maximize: [mean(meta_feature₁(train))/mean(meta_feature₁(test)), ..., mean(meta_featureₙ(train))/mean(meta_featureₙ(test))]
subject to: |test_set| = α × |dataset|
```

Where meta-features include, for example:

ℹ️ **Info-theory:**
- **Attribute Entropy** (`attr_ent`): Measures feature distribution complexity
- **Class Concentration** (`class_conc`): Quantifies class imbalance
- **Mutual Information** (`mut_inf`): Captures feature-target relationships
- **Interquartile Range** (`iq_range`): Describes distribution spread

🔢 **Statistical:**
- **Joint Entropy** (`joint_ent`): Measures overall dataset complexity
- **Kurtosis** (`kurtosis`): Measures distribution tail heaviness
- **Eigenvalues** (`eigenvalues`): Captures data structure variance

### Evolutionary Algorithm

The optimization uses a genetic algorithm with:
- **Population**: Lists of indices representing test set assignments
- **Fitness**: Ratio-based distance between meta-feature values (train/test ratios)
- **Selection**: NSGA-II multi-objective selection
- **Crossover**: Index exchange between individuals while maintaining test set size
- **Mutation**: Index replacement with available indices from train set

### Approach Diagram

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '10px' }}}%%
graph TD
    A[Input Data  
X, y] --> C[Evolutionary  
Optimization]
    subgraph Meta-Features["

Meta-Features

"]
        D["Statistical,  
Info-theory"]
    end
    subgraph Optimization["

Optimization

"]
        E["Population  
Train/Test Splits"] --> F["Meta-Feature  
Extraction"]
        F --> G["Fitness Function  
ratio = MF_train / MF_test"]
        G --> H["Selection"]
        H --> I["Crossover"]
        I --> J["Mutation"]
        J --> E
    end
    C --> E
    C --> K["Final Split"]
    K --> L["Train Set"] & M["Test Set"]
    style A fill:#761A29,stroke:#761A29,stroke-width:1px,color:#fff
    style C fill:#8A8F35,stroke:#8A8F35,stroke-width:1px,color:#fff
    style K fill:#287786,stroke:#287786,stroke-width:1px,color:#fff
    style L fill:#287786,stroke:#287786,stroke-width:1px,color:#fff
    style M fill:#287786,stroke:#287786,stroke-width:1px,color:#fff
    style D fill:#66B8C8,stroke:#66B8C8,stroke-width:1px,color:#fff
    style E fill:#DBA494,stroke:#DBA494,stroke-width:1px,color:#fff
    style F fill:#9FB88E,stroke:#9FB88E,stroke-width:1px,color:#fff
    style G fill:#DBA494,stroke:#DBA494,stroke-width:1px,color:#fff
    style H fill:#9FB88E,stroke:#9FB88E,stroke-width:1px,color:#fff
    style I fill:#9FB88E,stroke:#9FB88E,stroke-width:1px,color:#fff
    style J fill:#9FB88E,stroke:#9FB88E,stroke-width:1px,color:#fff
    linkStyle default stroke:#287786,stroke-width:1.5px
```

### 📊 Experimental Results: Train/Test Split Analysis

📌 **Bold** indicates best result in category

| Split type | Dataset | LR | XGB | IRM | DRO |
|--------|---------|----|----|-----|-----|
| **Random Split**| taxi | 0.752 ± 0.01 | 0.778 ± 0.01 | 0.790 ± 0.02 | 0.712 ± 0.02 |
| **Class_conc** | taxi | **0.526** ± 0.10 | **0.592** ± 0.07 | **0.773** ± 0.10 | **0.505** ± 0.10 |
| **Random Split** | electricity | 0.798 ± 0.00 | 0.832 ± 0.00 | 0.813 ± 0.01 | 0.814 ± 0.02 |
| **Mut_inf** | electricity | **0.735** ± 0.02 | **0.749** ± 0.01 | **0.795** ± 0.02 | **0.766** ± 0.01 |

## 2️⃣ Synthetic Data Generation

The synthetic generation approach formulates data creation as an optimization problem:

```
minimize: ||meta_features(synthetic) - meta_features(target)||₂
subject to: synthetic ∈ feasible_space(source)
```

### Evolutionary Algorithm

The optimization uses a genetic algorithm with:
- **Population**: Samples generated by Forest diffusion from source data
- **Fitness**: Euclidean distance between synthetic and target meta-feature vectors
- **Selection**: NSGA-III multi-objective selection with reference points
- **Crossover**: Row-wise or column-wise exchange between data matrices
- **Mutation**: Multiple strategies: noise addition, distribution sampling, or covariance-based generation

### Approach Diagram

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '10px' }}}%%
graph TD
    A[Source Data] --> C[Forest Diffusion  
Model Training]
    B[Target Data] --> D[Target Meta-Features  
Extraction]
    subgraph Meta-Features["

Meta-Features

"]
        E["Statistical,  
Info-theory"]
    end
    subgraph Evolutionary Process["

Optimization

"]
        F["Population  
Synthetic Data"] --> G["Fitness Function  
||MF_synthetic - MF_target||"]
        G --> H["Selection"]
        H --> I["Crossover"]
        I --> J["Mutation  
Noise/Distribution/Covariance"]
        J --> F
    end
    C --> F
    D --> E
    G --> K["Best Synthetic Data"]
    style A fill:#761A29,stroke:#761A29,stroke-width:1px,color:#fff
    style B fill:#761A29,stroke:#761A29,stroke-width:1px,color:#fff
    style C fill:#8A8F35,stroke:#8A8F35,stroke-width:1px,color:#fff
    style D fill:#8A8F35,stroke:#8A8F35,stroke-width:1px,color:#fff
    style K fill:#287786,stroke:#287786,stroke-width:1px,color:#fff
    style E fill:#66B8C8,stroke:#66B8C8,stroke-width:1px,color:#fff
    style F fill:#DBA494,stroke:#DBA494,stroke-width:1px,color:#fff
    style G fill:#DBA494,stroke:#DBA494,stroke-width:1px,color:#fff
    style H fill:#9FB88E,stroke:#9FB88E,stroke-width:1px,color:#fff
    style I fill:#9FB88E,stroke:#9FB88E,stroke-width:1px,color:#fff
    style J fill:#9FB88E,stroke:#9FB88E,stroke-width:1px,color:#fff
    linkStyle default stroke:#287786,stroke-width:1.5px
```

### 📊 Experimental Results: Synthetic Data Generation Analysis

Performance on synthetic data generated with optimized meta-features:

| Dataset | LR | XGB | DRO | IRM |
|---------|----|-----|-----|-----|
| **electricity** (mut-inf, class-conc, iq-range) | 0.613 ± 0.08 | 0.641 ± 0.09 | 0.587 ± 0.08 | 0.613 ± 0.08 |
| **electricity** (mut-inf, class-conc) | 0.611 ± 0.01 | 0.625 ± 0.01 | 0.589 ± 0.01 | 0.632 ± 0.02 |

## 3️⃣ CTGAN-Based Distribution Shifter

The `ctgan_shifter` module implements a latent-space distribution shifting approach using CTGAN. It learns a differentiable shift in the latent representation of tabular data to generate controlled out-of-distribution samples.

### Key Components

- **`ctgan_shifter/src/shifter.py`**: Core `Shifter` model — a differentiable latent-space transformation
- **`ctgan_shifter/src/ctgan_adapter.py`**: Adapter connecting the CTGAN generator with the Shifter
- **`ctgan_shifter/src/differentiable_mfe.py`**: Differentiable meta-feature extractor for gradient-based optimization
- **`ctgan_shifter/preprocessing/tab_preprocessing.py`**: Tabular data preprocessing utilities
- **`ctgan_shifter/external/ctgan_repo/`**: CTGAN implementation (synthesizers, data transformer, sampler)
- **`ctgan_shifter/example/`**: Example artifacts for the Electricity dataset demo

### How It Works

1. A CTGAN model is trained on source tabular data
2. A `Shifter` network learns a latent-space transformation to match target meta-feature values
3. The shifted CTGAN generates synthetic OOD samples with controlled statistical properties
4. The generated data is used for robustness evaluation

For full details, see [`ctgan_shifter/README.md`](ctgan_shifter/README.md).

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ITMO-NSS-team/OOD_Tab_Evaluation.git
cd OOD_Tab_Evaluation
```

## 🔧 Reproducing Experiments

### 1. Download Datasets

The repository includes several tabular datasets with known distributional shifts:

```
data/
├── electricity_source.csv  # Source domain data
├── electricity_target.csv  # Target domain data
...
├── taxi_source.csv
└── taxi_target.csv
```

### 2. Run Meta-Feature Splitting

```python
from mfs_split.mfs_split import run_split
import pandas as pd

# Load your data
data = pd.read_csv('your_data.csv')

# Run split optimization
run_split(
    file=data,
    target_column_name='target',       # Your target column name
    file_prefix_name='split_by_class_conc',  # Output file prefix
    meta_features=['class_conc'],      # Meta-feature to optimize
    population_size=50,
    generations=300
)
```

### 3. Run Synthetic Data Generation

```python
from mfs_split.mfs_synthetic import run_shift_convergence_experiment

# Generate synthetic data
results = run_shift_convergence_experiment(
    meta_features=['class_conc', 'mut_inf'],  # Meta-features to match
    mutation_type='all',                       # Mutation strategy
    n_samples=dataset_length,                 # Number of samples to generate
    generations=200,
    source_file='data/source.csv',
    target_file='data/target.csv'
)
```

### 4. Run CTGAN Shifter

```python
from ctgan_shifter.src.ctgan_adapter import CTGANAdapter
from ctgan_shifter.src.shifter import Shifter

# Train CTGAN on source data
adapter = CTGANAdapter()
adapter.fit(source_data)

# Apply distribution shift
shifter = Shifter(adapter)
shifted_data = shifter.generate(target_meta_features)
```

## 📁 Repository Structure

```
OOD_Tab_Evaluation/
├── mfs_split/                     # Meta-feature splitting algorithms
│   ├── mfs_split.py               # Advanced DEAP-based implementation
│   └── mfs_synthetic.py           # Synthetic data generation
├── robust_models/                 # Robust model implementations
│   ├── IRM_model/                 # Invariant Risk Minimization
│   └── DRO_model/                 # Distributionally Robust Optimization
├── baselines/                     # Baseline methods
│   └── worst_case_subpopulation.py
├── ctgan_shifter/                 # CTGAN-based distribution shifter
│   ├── src/                       # Core shifter modules
│   │   ├── shifter.py             # Latent-space Shifter model
│   │   ├── ctgan_adapter.py       # CTGAN-Shifter adapter
│   │   └── differentiable_mfe.py  # Differentiable meta-feature extractor
│   ├── preprocessing/             # Data preprocessing
│   │   └── tab_preprocessing.py
│   ├── external/ctgan_repo/       # CTGAN implementation
│   │   └── ctgan/                 # Synthesizers, transformer, sampler
│   ├── example/                   # Demo artifacts (Electricity dataset)
│   └── README.md                  # Shifter module documentation
├── data/                          # Dataset files
├── experiments/                   # Experiment scripts and results
└── README.md                      # This file
```

## 📚 Dependencies

- `numpy>=1.21.0`
- `pandas>=1.3.0`
- `scikit-learn>=1.0.0`
- `torch>=1.9.0`
- `xgboost>=1.5.0`
- `deap>=1.3.0`
- `pymfe>=0.4.0`
- `matplotlib>=3.5.0`
- `ctgan>=0.7.0`

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{Deeva2025,
  author = {Deeva, Irina and Amerkhanova, Nargiza},
  title = {Evaluating robustness of tabular models under meta-features based shifts},
  booktitle = {Proceedings of the 39th Annual Conference on Neural Information Processing Systems (NeurIPS '25)},
  year = {2025},
  address = {San Diego, USA},
}
```

## 📞 Contact

For questions or feedback, please open an issue on GitHub or contact the maintainers.
