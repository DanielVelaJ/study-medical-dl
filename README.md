# Medical Deep Learning Studies

This repository contains mini-study sessions for medical deep learning research. Each study is self-contained and designed for reproducible research.

## 📁 Study Sessions

### `01_roco_mmd/` - Medical CLIP Models MMD Analysis

Replicates Figure 4 from Baghbanzadeh et al.'s "Open-PMC-18M" paper ([arXiv:2506.02738](https://arxiv.org/abs/2506.02738)). Uses **Maximum Mean Discrepancy (MMD)** to demonstrate that different CLIP models create statistically different embedding distributions on biomedical images.

**Key Question**: Do medical-specialist CLIP models create significantly different embedding distributions compared to general-purpose models?

**Quick Start**:
```bash
cd 01_roco_mmd
# Open notebook.ipynb and run all cells
# Everything is self-contained with automatic setup
```

**What it does**:
- Compares [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) (medical specialist) vs [OpenAI CLIP](https://github.com/mlfoundations/open_clip) (general purpose)
- Encodes 200 ROCO medical images with both models  
- Computes MMD with RBF kernel and runs 100-permutation statistical test
- Visualizes results with t-SNE and significance testing

**Expected output**: Statistically significant differences (p < 0.01) between embedding distributions

---

## 🛠️ Setup

Each study is designed to be self-contained. We recommend using conda for environment management:

### Option 1: Conda Environment (Recommended)
```bash
# Create and activate environment with binary dependencies (PyTorch, CUDA)
conda env create -f environment.yml
conda activate biomedical-clip

# Install remaining packages with pip
pip install -r requirements.txt

# Navigate to study folder
cd 01_roco_mmd/
# Open notebook.ipynb and run all cells
```

### Option 2: Manual Installation
```bash
# Prerequisites for 01_roco_mmd
pip install torch torchvision open-clip-torch datasets scikit-learn matplotlib seaborn tqdm

# Navigate to study folder
cd 01_roco_mmd/
# Open the notebook and run all cells
```

**Note**: The notebook handles data downloading, model loading, and analysis automatically.

## 📋 Planned Studies

- `02_clip_medical/` - Medical CLIP fine-tuning strategies
- `03_multimodal_fusion/` - Cross-modal attention mechanisms  
- `04_segmentation_medical/` - Medical image segmentation benchmarks
- `05_gan_medical/` - Medical image synthesis evaluation

## 📚 References

Core papers driving these studies:
- **OPEN-PMC-18M**: Baghbanzadeh et al., [arXiv 2506.02738](https://arxiv.org/abs/2506.02738)
- **BiomedCLIP**: Zhang et al., ["BiomedCLIP: A Multimodal Biomedical Foundation Model"](https://arxiv.org/abs/2303.00915)
- **OpenAI CLIP**: Radford et al., ["Learning Transferable Visual Representations"](https://arxiv.org/abs/2103.00020)
- **MMD Testing**: Gretton et al., ["A Kernel Two-Sample Test"](https://jmlr.org/papers/v13/gretton12a.html)

## 🔬 Study Design Philosophy

Each study session follows these principles:
- **Self-contained**: All code in a single notebook with automatic setup
- **Reproducible**: Detailed documentation and fallback mechanisms
- **Educational**: Step-by-step explanations with academic references
- **Practical**: Ready-to-run implementations with real datasets

## 📄 License

MIT License - Each study folder may have its own specific license requirements.