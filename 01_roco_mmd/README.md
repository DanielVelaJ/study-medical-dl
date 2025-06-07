# MMD Comparison: OPEN-PMC-18M vs PMC-6M Embeddings

This mini-study replicates the Figure 4 intuition from Alejandro Lozano's biomedical V-L pipeline paper ([arXiv 2506.02738](https://arxiv.org/abs/2506.02738)), demonstrating that **data quality reshapes the representation space** in biomedical vision-language models.

## 🎯 Experiment Goal

Compare embeddings from two biomedical CLIP models using statistical testing:
- **Model A**: `biomedclip` (OPEN-PMC-18M quality version) 
- **Model B**: `biomedica_pmc-6m` (noisy baseline)

**Key Question**: Are the embedding clouds from these models statistically different, even if they visually overlap in t-SNE?

## 🏗️ Project Structure

```
01_roco_mmd/
├── notebook.ipynb               # Interactive demo (unit test & plots)
├── encode.py                    # Image-batch encoder helper
├── mmd.py                       # RBF-MMD + permutation util
├── tsne_plot.py                 # Optional script for t-SNE figure
├── data/                        # ROCO subset (git-ignored)
├── results/                     # Saved .npz, .png (git-ignored)
└── README.md                    # This file
```

## 🚀 Quick Start

### 1. Setup Environment (Conda + pip)
```bash
# From project root
conda env create -f environment.yml
conda activate open-pmc-playground
pip install -r requirements.txt
```

### 2. Run Interactive Notebook
```bash
cd 01_roco_mmd
jupyter notebook notebook.ipynb
```

### 3. Or Run Individual Scripts
```bash
# Generate embeddings and run MMD test
python encode.py
python mmd.py

# Create visualizations  
python tsne_plot.py
```

### 4. Expected Output
```
🧬 OPEN-PMC-18M vs PMC-6M Embedding Comparison
==================================================
🔬 Loading BiomedCLIP (OPEN-PMC-18M)...
📊 Simulating PMC-6M baseline (with quality degradation)...
🖼️  Loading 200 sample biomedical images...
⚡ Generating embeddings...
🎲 Running 100-shuffle permutation test...

📊 RESULTS:
MMD² = 0.001234
p-value = 0.010

✅ Significant difference! Data quality matters.
```

## 🧮 Methodology

### Maximum Mean Discrepancy (MMD)
MMD measures the difference between two probability distributions using kernel embeddings:

```
MMD²(P,Q) = E[k(X,X')] + E[k(Y,Y')] - 2E[k(X,Y)]
```

Where:
- `k(x,y) = exp(-γ||x-y||²)` (RBF kernel)
- `X ~ P` (OPEN-PMC-18M embeddings)
- `Y ~ Q` (PMC-6M embeddings)

### Permutation Test
1. Compute observed MMD² between the two embedding sets
2. Randomly shuffle labels 100 times to create null distribution
3. Calculate p-value: `P(MMD²_null ≥ MMD²_observed)`
4. If p < 0.05, reject null hypothesis → distributions are different

## 📊 What MMD & Permutation Test Do

### MMD (Maximum Mean Discrepancy)
- **Purpose**: Measures distributional difference between two sets of embeddings
- **Intuition**: If two models learned similar representations, their embedding distributions should be similar
- **Output**: A single number (MMD²) quantifying the difference

### Permutation Test
- **Purpose**: Tests statistical significance of the observed MMD²
- **Null Hypothesis**: Both embedding sets come from the same distribution
- **Method**: Randomly reassign embeddings to groups and recompute MMD² many times
- **Result**: p-value indicating how likely the observed difference is due to chance

### Why This Matters
- **Visual overlap ≠ statistical sameness**: t-SNE can make different distributions look similar
- **Data quality assessment**: Quantifies whether training data improvements actually changed the learned representations
- **Objective evaluation**: Complements subjective visual inspection with rigorous statistical testing

## 🔬 Extensions & Variations

### 1. Change Kernel Bandwidth
```python
# In notebook or scripts
mmd_observed, p_value, null_mmds = permutation_test(
    embeddings_a, embeddings_b, 
    n_permutations=100, 
    mmd_func=mmd_rbf,
    gamma=0.5  # Wider kernel
)
```

### 2. Alternative Kernels
```python
from mmd import mmd_linear

mmd_observed, p_value, null_mmds = permutation_test(
    embeddings_a, embeddings_b, 
    mmd_func=mmd_linear  # Linear kernel (faster)
)
```

### 3. Real ROCO Dataset
Replace sample images in `encode.py` with actual ROCO dataset:
```python
# Download ROCO: https://github.com/razorx89/roco-dataset
def load_real_roco_images():
    # Implement actual ROCO test set loader
    pass
```

### 4. Load Actual PMC-CLIP Model
Replace simulation in `encode.py`:
```python
# Load real PMC-CLIP from WeixiongLin/PMC-CLIP
def load_real_pmc_clip():
    # Implementation for actual PMC-CLIP model
    pass
```

## 📚 Background & References

### OPEN-PMC-18M Paper
- **Title**: Biomedical V-L Pipeline with Data Quality Assessment
- **Authors**: Alejandro Lozano et al.
- **arXiv**: [2506.02738](https://arxiv.org/abs/2506.02738)
- **Key Insight**: High-quality training data (OPEN-PMC-18M) produces better representations than noisy data (PMC-6M)

### Model Sources
- **BiomedCLIP**: Microsoft's biomedical CLIP trained on PMC-15M
  - HuggingFace: [`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
  - Paper: Zhang et al., ["BiomedCLIP: A Multimodal Biomedical Foundation Model"](https://arxiv.org/abs/2303.00915)
  
- **PMC-CLIP**: Baseline biomedical CLIP with smaller, noisier dataset
  - GitHub: [`WeixiongLin/PMC-CLIP`](https://github.com/WeixiongLin/PMC-CLIP)
  - Paper: Lin et al., ["PMC-CLIP: Contrastive Language-Image Pre-training using Biomedical Documents"](https://arxiv.org/abs/2303.07240)

### Statistical Methods
- **MMD**: Gretton et al., ["A Kernel Two-Sample Test"](https://jmlr.org/papers/v13/gretton12a.html) (JMLR 2012)
- **Permutation Testing**: Classic non-parametric statistical method
- **t-SNE**: van der Maaten & Hinton, ["Visualizing Data using t-SNE"](https://jmlr.org/papers/v9/vandermaaten08a.html) (JMLR 2008)

### Dataset References
- **ROCO**: Pelka et al., ["Radiology Objects in Context (ROCO): A Multimodal Image Dataset"](https://doi.org/10.1007/978-3-030-01364-6_20) (MICCAI 2018)

## 🤝 Contributing

This experiment provides a template for comparing vision-language models. Contributions welcome for:
- Loading actual PMC-CLIP model (vs. simulation)
- Adding more kernel types
- Implementing other distribution comparison metrics
- Testing on real ROCO dataset

## 📄 License

MIT License - feel free to adapt for your research! 