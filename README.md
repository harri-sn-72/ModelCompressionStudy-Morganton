# omparative Evaluation of Neural Network Compression Techniques Across Cloud GPU, NPU, and CPU Platforms

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**A systematic evaluation of neural network compression techniques across cloud GPU, modern laptop NPU, and legacy desktop CPU platforms.**

---

## Overview

This repository contains the complete code, data, and figures for the research paper: **"Comparative Evaluation of Neural Network Compression Techniques Across Cloud GPU, NPU, and CPU Platforms"** 

### Key Findings

**Modern laptop NPUs achieve 3× faster inference than cloud GPUs**
- Intel Core Ultra 5: 3.5× speedup (MNIST), 3.0× speedup (CIFAR-10)
- Even legacy 2017 CPUs matched cloud GPU performance

**Quantization amplifies NPU advantages**
- NPU hardware shows 26% larger quantization speedup vs GPU
- INT8 quantization: 3.5× size reduction, maintained accuracy, 1.87× speedup on NPU

**Compression enables 10× speedup on legacy hardware**
- Knowledge-distilled student models: 10.48× faster on desktop CPU
- Demonstrates viability of older hardware with appropriate compression

**Task complexity determines pruning tolerance**
- MNIST: Safe up to 70% pruning
- CIFAR-10: Catastrophic degradation at 90% sparsity (19.2% accuracy drop)

---

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


## 🙏 Acknowledgments

- **Google Colab** for providing free cloud GPU resources
- **PyTorch team** for the excellent deep learning framework
- **Intel** for Core Ultra 5 NPU hardware and developer tools
- **MNIST and CIFAR-10 dataset creators** for benchmark datasets
- **Open-source ML community** for tools and inspiration
- **Special thanks to Claude (Anthropic)** for assistance with code development and debugging throughout this research project.

---

## 🐛 Known Issues / Limitations

1. **Intel Arc GPU not utilized**: Experiments ran on CPU backend with potential NPU acceleration. Direct XPU usage may show different results.

2. **Pruned model sizes unchanged**: PyTorch standard serialization doesn't implement sparse storage. File sizes remain constant despite reduced operations.

3. **Limited to simple CNNs**: Results may not generalize to larger architectures (ResNet, VGG, Transformers).

4. **No power measurements**: Energy efficiency not measured, only inference latency.

5. **Single-image inference**: Batch processing performance not evaluated.

---
