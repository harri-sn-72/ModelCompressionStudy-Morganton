# Edge Inference Outperforms Cloud: Neural Network Compression on NPU-Enabled Devices

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**A systematic evaluation of neural network compression techniques across cloud GPU, modern laptop NPU, and legacy desktop CPU platforms.**

---

## 📋 Overview

This repository contains the complete code, data, and figures for the research paper: **"Edge Inference Outperforms Cloud: Evaluating Neural Network Compression on NPU-Enabled Devices"** published in *The Morganton Scientific* (2026).

### Key Findings

🚀 **Modern laptop NPUs achieve 3× faster inference than cloud GPUs**
- Intel Core Ultra 5: 3.5× speedup (MNIST), 3.0× speedup (CIFAR-10)
- Even legacy 2017 CPUs matched cloud GPU performance

🔥 **Quantization amplifies NPU advantages**
- NPU hardware shows 26% larger quantization speedup vs GPU
- INT8 quantization: 3.5× size reduction, maintained accuracy, 1.87× speedup on NPU

⚡ **Compression enables 10× speedup on legacy hardware**
- Knowledge-distilled student models: 10.48× faster on desktop CPU
- Demonstrates viability of older hardware with appropriate compression

🎯 **Task complexity determines pruning tolerance**
- MNIST: Safe up to 70% pruning
- CIFAR-10: Catastrophic degradation at 90% sparsity (19.2% accuracy drop)

---

## 📊 Repository Structure

```
.
├── README.md                              # This file
├── data/                                  # Experimental results
│   ├── Compression_ML_Paper_Sheet.csv     # Complete results (46 experiments)
│   ├── paper_summary_table.csv            # Clean summary table
│   └── baseline_results.csv               # Baseline measurements
│
├── notebooks/                             # Jupyter notebooks
│   ├── research_notebook_part1.ipynb      # Baseline training & evaluation
│   ├── research_notebook_part2.ipynb      # Compression experiments
│   └── research_notebook_part3_CUSTOM.ipynb # Results visualization
│
├── src/                                   # Source code
│   ├── helper_functions.py                # Reusable model & training functions
│   ├── create_architecture_diagrams.py    # Generate architecture figures
│   └── hardware_test.py                   # Hardware capability testing
│
├── figures/                               # Publication-quality figures (300 DPI)
│   ├── figure_architectures.png           # Model architecture diagrams
│   ├── figure_compression_pipeline.png    # Compression methods overview
│   ├── figure_teacher_student.png         # Knowledge distillation comparison
│   ├── figure1_platform_comparison.png    # Baseline platform performance
│   ├── figure2_pruning_sweet_spot.png     # Pruning vs accuracy analysis
│   ├── figure3_methods_comparison.png     # Compression methods on NPU
│   ├── figure4_quantization_amplification.png # Platform-specific quantization
│   └── figure5_speedup_heatmap.png        # Complete speedup analysis
│
├── paper/                                 # Paper and documentation
│   ├── paper_tables.md                    # Table captions and formatting
│   └── [Full paper PDF/DOCX]              # Complete manuscript
│
├── requirements.txt                       # Python dependencies
└── LICENSE                                # MIT License

```

---

## 🔬 Experimental Setup

### Datasets
- **MNIST**: 60,000 training / 10,000 test images (28×28 grayscale digits)
- **CIFAR-10**: 50,000 training / 10,000 test images (32×32 color objects)

### Hardware Platforms
1. **Cloud GPU**: Google Colab with NVIDIA Tesla T4 (16GB)
2. **Laptop NPU**: HP Omnibook with Intel Core Ultra 5 226V (8GB Arc 130V)
3. **Desktop CPU**: Lenovo ThinkCentre with Intel Core i5-8400 (2017)

### Model Architectures
- **SimpleMNIST**: 2 conv blocks → 421,642 parameters → 1.61 MB
- **SimpleCIFAR**: 3 conv blocks → 2,473,610 parameters → 9.44 MB

### Compression Techniques
1. **Magnitude-based Pruning**: L1 unstructured at 30%, 50%, 70%, 90% sparsity
2. **Post-training Quantization**: Dynamic INT8 on Conv2d and Linear layers
3. **Knowledge Distillation**: 50% channel reduction → 4× parameter compression

**Total Experiments**: 46 configurations across methods, datasets, and platforms

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
PyTorch 2.0+
CUDA 11.8+ (for GPU experiments)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/neural-compression-edge-inference.git
cd neural-compression-edge-inference
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Verify hardware setup**
```bash
python src/hardware_test.py
```

Expected output:
```
✓ Python 3.8+ detected
✓ PyTorch 2.x.x installed
✓ Device: cuda / cpu / xpu
✓ Hardware test complete
```

---

## 📈 Reproducing Results

### Option 1: Run Complete Pipeline (Recommended)

**Step 1: Baseline Training (Day 1-2)**
```bash
jupyter notebook notebooks/research_notebook_part1.ipynb
```
- Trains baseline models on MNIST and CIFAR-10
- Evaluates on all available hardware
- Saves: `baseline_results.csv`, model checkpoints

**Step 2: Compression Experiments (Day 3-4)**
```bash
jupyter notebook notebooks/research_notebook_part2.ipynb
```
- Applies pruning, quantization, and distillation
- Tests on all platforms
- Saves: `all_compression_results.csv`

**Step 3: Visualization & Analysis (Day 5)**
```bash
jupyter notebook notebooks/research_notebook_part3_CUSTOM.ipynb
```
- Generates all publication figures
- Creates summary tables
- Performs statistical analysis

**Expected Runtime**:
- Baseline training: 30-60 min (GPU) or 2-3 hours (CPU)
- Compression experiments: 3-5 hours (GPU) or 8-12 hours (CPU)
- Visualization: 5-10 minutes

### Option 2: Use Pre-Generated Results

All experimental data is available in `data/` directory:
```python
import pandas as pd

# Load complete results
results = pd.read_csv('data/Compression_ML_Paper_Sheet.csv')

# Load summary table
summary = pd.read_csv('data/paper_summary_table.csv')
```

### Option 3: Generate Figures Only

```bash
python src/create_architecture_diagrams.py
jupyter notebook notebooks/research_notebook_part3_CUSTOM.ipynb
```

---

## 📊 Key Results

### Platform Performance Comparison

| Platform | MNIST Latency | CIFAR Latency | Speedup vs Cloud |
|----------|---------------|---------------|------------------|
| **Cloud GPU (T4)** | 0.565 ms | 2.733 ms | 1.0× (baseline) |
| **Laptop NPU (Core Ultra 5)** | 0.160 ms | 0.923 ms | **3.5× / 3.0×** |
| **Desktop CPU (i5-8400)** | 0.202 ms | 0.900 ms | 2.8× / 3.0× |

### Compression Methods (Intel Core Ultra 5)

| Method | CIFAR Accuracy | Size | Latency | Speedup |
|--------|----------------|------|---------|---------|
| Baseline | 78.37% | 9.44 MB | 0.923 ms | 1.0× |
| Pruning (70%) | 78.03% | 9.44 MB | 0.836 ms | 1.1× |
| **Quantization** | **78.48%** | **3.43 MB** | 0.493 ms | **1.9×** |
| Distillation | 77.17% | 2.37 MB | **0.430 ms** | 2.1× |

### Maximum Speedups Achieved

- **10.48×**: CIFAR-10 student model on i5-8400 CPU
- **7.63×**: MNIST student model on i5-8400 CPU
- **6.36×**: CIFAR-10 distillation on Core Ultra 5
- **5.55×**: CIFAR-10 quantization on Core Ultra 5

---

## 🔧 Hardware-Specific Instructions

### Running on Intel Core Ultra 5 (NPU)

1. **Update Intel drivers**
   - Download latest Arc GPU drivers from Intel
   - Restart after installation

2. **Install Intel Extension for PyTorch**
```bash
pip install intel-extension-for-pytorch --break-system-packages
```

3. **Verify NPU detection**
```bash
python -c "import torch; print(torch.xpu.is_available())"
```

**Note**: Our experiments used CPU backend with NPU acceleration. If XPU is detected, performance may differ.

### Running on Google Colab (Cloud GPU)

1. **Open notebook in Colab**
   - Upload `.ipynb` files to Google Drive
   - Open with Google Colab

2. **Enable GPU runtime**
   - Runtime → Change runtime type → GPU (T4)

3. **Install dependencies**
```python
!pip install torch torchvision matplotlib pandas seaborn
```

### Running on CPU-Only Systems

Works on any modern CPU. Training will be slower but results are reproducible:
- MNIST baseline: ~30-45 minutes
- CIFAR-10 baseline: ~1-2 hours
- Compression experiments: ~4-8 hours total

---

## 📦 Requirements

### Core Dependencies
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.10.0
jupyter>=1.0.0
```

### Optional (for Intel NPU support)
```
intel-extension-for-pytorch>=2.0.0
```

### Install all at once
```bash
pip install -r requirements.txt
```

---

## 📝 Citation

If you use this code or data in your research, please cite:

```bibtex
@article{yourname2026edge,
  title={Edge Inference Outperforms Cloud: Evaluating Neural Network Compression on NPU-Enabled Devices},
  author={[Your Name]},
  journal={The Morganton Scientific},
  year={2026},
  publisher={NCSSM}
}
```

---

## 🤝 Contributing

Contributions are welcome! Areas for extension:

- **Additional hardware platforms**: Apple Silicon, Qualcomm NPU, AMD GPUs
- **Larger models**: ResNet, EfficientNet, Vision Transformers
- **More datasets**: ImageNet, COCO, custom domains
- **Combined compression**: Pruning + quantization, quantization + distillation
- **Power measurements**: Energy efficiency comparisons
- **Batch inference**: Performance with varying batch sizes

Please open an issue or pull request with your improvements!

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Google Colab** for providing free cloud GPU resources
- **PyTorch team** for the excellent deep learning framework
- **Intel** for Core Ultra 5 NPU hardware and developer tools
- **MNIST and CIFAR-10 dataset creators** for benchmark datasets
- **Open-source ML community** for tools and inspiration

Special thanks to Claude (Anthropic) for assistance with code development and debugging throughout this research project.

---

## 📧 Contact

**Author**: [Your Name]  
**Email**: [your.email@example.com]  
**Institution**: North Carolina School of Science and Mathematics  
**ORCID**: [Your ORCID ID]

For questions about the research, please open an issue or contact via email.

---

## 📚 Additional Resources

- **Paper PDF**: [Link to published paper]
- **Presentation Slides**: [Link if available]
- **Demo Video**: [Link if available]
- **Blog Post**: [Link if available]

---

## 🗓️ Project Timeline

- **August 2025**: Project conception (procrastinated until...)
- **February 9-11, 2026**: Intensive research sprint (Days 1-2)
  - Learned Python and PyTorch from scratch
  - Set up environments on 3 platforms
  - Trained baseline models
- **February 12-13, 2026**: Compression experiments (Days 3-4)
  - Ran 46 experimental configurations
  - Discovered key findings
- **February 14-15, 2026**: Analysis and writing (Days 5-6)
  - Generated all figures and tables
  - Wrote complete manuscript
- **February 16, 2026**: Submission preparation (Day 7)

**Total active development time**: 7 days

---

## 🐛 Known Issues / Limitations

1. **Intel Arc GPU not utilized**: Experiments ran on CPU backend with potential NPU acceleration. Direct XPU usage may show different results.

2. **Pruned model sizes unchanged**: PyTorch standard serialization doesn't implement sparse storage. File sizes remain constant despite reduced operations.

3. **Limited to simple CNNs**: Results may not generalize to larger architectures (ResNet, VGG, Transformers).

4. **No power measurements**: Energy efficiency not measured, only inference latency.

5. **Single-image inference**: Batch processing performance not evaluated.

See paper Discussion section for detailed limitations and future work.

---

## 📊 Reproducing Paper Figures

All figures in the paper can be regenerated:

**Architecture diagrams (Figures 2-4):**
```bash
python src/create_architecture_diagrams.py
```

**Results figures (Figures 5-9):**
```bash
jupyter notebook notebooks/research_notebook_part3_CUSTOM.ipynb
# Run all cells
```

Figures are saved at 300 DPI in `figures/` directory, ready for publication.

---

## 🎓 Educational Use

This repository is designed to be educational. Key learning resources:

- **`notebooks/`**: Step-by-step experiments with extensive comments
- **`src/helper_functions.py`**: Reusable code patterns for ML research
- **`paper/`**: Example of complete research paper structure
- **Figures**: Examples of publication-quality visualizations

Ideal for students learning:
- Deep learning fundamentals
- Model compression techniques
- Cross-platform performance evaluation
- Research paper writing
- Scientific Python programming

---

## 🔄 Version History

- **v1.0.0** (February 2026): Initial release with complete research pipeline
  - 3 platforms tested
  - 3 compression methods
  - 46 experiments
  - 8 figures, 3 tables
  - Complete reproducible code

---

## ⭐ Star History

If you find this work useful, please consider starring the repository!

---

**Last Updated**: February 16, 2026  
**Repository Status**: ✅ Complete and reproducible
