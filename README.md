# Medical Deep Learning Studies

This repository contains mini-study sessions for medical deep learning research.

## 📁 Study Sessions

### `01_roco_mmd/` - MMD Comparison: OPEN-PMC-18M vs PMC-6M Embeddings

Replicates Figure 4 intuition from Alejandro Lozano's biomedical V-L pipeline paper ([arXiv 2506.02738](https://arxiv.org/abs/2506.02738)). Demonstrates that **data quality reshapes the representation space** in biomedical vision-language models using Maximum Mean Discrepancy (MMD) and permutation testing.

**Key Question**: Are embedding clouds from high-quality vs noisy training data statistically different?

**Quick Start**:
```bash
cd 01_roco_mmd
poetry install
poetry run python main.py
```

**What it does**:
- Loads [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) (OPEN-PMC-18M) and simulates [PMC-6M baseline](https://github.com/WeixiongLin/PMC-CLIP)
- Encodes 200 ROCO test images with both models  
- Computes MMD² and runs 100-shuffle permutation test
- Visualizes results with t-SNE + statistical significance

**Expected output**: `MMD² = 0.001234, p-value = 0.010` → Significant difference!

---

## 🛠️ Development Setup

This repo uses Poetry for dependency management and follows absolute import patterns:

```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Navigate to any study folder and install
cd 01_roco_mmd/
poetry install
poetry run python main.py
```

## 📋 Planned Studies

- `02_clip_medical/` - Medical CLIP fine-tuning strategies
- `03_multimodal_fusion/` - Cross-modal attention mechanisms  
- `04_segmentation_medical/` - Medical image segmentation benchmarks
- `05_gan_medical/` - Medical image synthesis evaluation

## 📚 References

Core papers driving these studies:
- **OPEN-PMC-18M**: Lozano et al., [arXiv 2506.02738](https://arxiv.org/abs/2506.02738)
- **BiomedCLIP**: Zhang et al., ["BiomedCLIP: A Multimodal Biomedical Foundation Model"](https://arxiv.org/abs/2303.00915)
- **PMC-CLIP**: Lin et al., ["PMC-CLIP: Contrastive Language-Image Pre-training using Biomedical Documents"](https://arxiv.org/abs/2303.07240)
- **MMD Testing**: Gretton et al., ["A Kernel Two-Sample Test"](https://jmlr.org/papers/v13/gretton12a.html)

## 🔬 Future Studies

Additional mini-studies will be added to explore various aspects of medical deep learning:
- Vision-language model evaluation
- Biomedical image analysis techniques
- Clinical data processing methods
- Multi-modal learning approaches

## 📄 License

MIT License - Each study folder may have its own specific license requirements.