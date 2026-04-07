# Akoya Histopathology Multimodal ML System

A complete, end-to-end machine learning project that reproduces and documents the Akoya Biosciences histopathology pipeline. The project moves through four architectural phases:

1. **Phase 1: CNN Baseline** — ResNet-50 fine-tuned for tile classification with focal loss
2. **Phase 2: Vision Transformer** — ViT-Base with FlashAttention and distributed training (DDP/FSDP)
3. **Phase 3: Multimodal VLM** — LLaVA-style vision-language model with LoRA fine-tuning
4. **Phase 4: Production Inference** — TensorRT INT8 optimization with dynamic batching

This is a portfolio/research reproduction project — not a production system. The codebase is designed to be clean, well-documented, and runnable on free GPU tiers (Google Colab, Kaggle).

## Project Structure

```
akoya-histopathology-ml/
├── README.md                        # This file
├── requirements.txt                 # All dependencies pinned
├── environment.yml                  # Conda environment file
├── .github/workflows/ci.yml        # GitHub Actions: lint + test on push
│
├── data/
│   ├── download_patches.py          # Synthetic dataset generator
│   ├── manifest.csv                 # Generated: filepath, label, split
│   ├── README.md                    # Dataset format documentation
│   └── sample/                      # 20 sample PNG tiles
│
├── notebooks/
│   ├── 01_data_exploration.ipynb    # EDA: class distribution, pixel stats
│   ├── 02_cnn_baseline.ipynb        # ResNet-50 training + evaluation
│   ├── 03_vit_training.ipynb        # ViT + FlashAttention (Session 2)
│   ├── 04_multimodal_llm.ipynb      # VLM architecture + LoRA (Session 3)
│   └── 05_inference_optimization.ipynb  # TensorRT INT8 (Session 4)
│
├── src/
│   ├── data/                        # Dataset, augmentations, DALI pipeline
│   ├── models/                      # ResNet-50, ViT, projection, VLM
│   ├── training/                    # Trainer, losses, metrics
│   ├── inference/                   # TRT export, calibration, serving
│   └── profiling/                   # PyTorch profiler, roofline analysis
│
├── configs/                         # YAML hyperparameter configs
├── scripts/                         # CLI training and inference scripts
├── tests/                           # Unit and integration tests
└── docs/                            # Architecture and guide documentation
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or with conda:
```bash
conda env create -f environment.yml
conda activate akoya-histo
```

### 2. Generate Synthetic Dataset

```bash
python data/download_patches.py --output_dir data/ --num_tiles 2000
```

Generates 2,000 synthetic 1024×1024 RGB PNG tiles across 4 classes with imbalanced distribution:
- **tumor** (15%): Dense purple-stained nuclei
- **immune** (20%): Scattered small dark circular nuclei
- **stroma** (55%): Fibrous pink texture
- **necrosis** (10%): Pale ghost-cell appearance

### 3. Train ResNet-50 Baseline

```bash
# GPU training
python scripts/train_resnet.py \
    --config configs/resnet_config.yaml \
    --data_dir data/ \
    --output_dir outputs/resnet/ \
    --device cuda

# CPU testing (2 epochs)
python scripts/train_resnet.py \
    --config configs/resnet_config.yaml \
    --data_dir data/ \
    --output_dir outputs/resnet/ \
    --device cpu \
    --epochs 2
```

### 4. Run Tests

```bash
python -m pytest tests/ -v
```

### 5. Explore Notebooks

```bash
jupyter notebook notebooks/
```

## Key Design Decisions

### Why Synthetic Data?
Real medical images require IRB approval. Synthetic data enables full reproducibility while validating the ML pipeline architecture.

### Why Focal Loss?
With 5.5:1 class imbalance (stroma vs. necrosis), standard cross-entropy is dominated by easy majority-class examples. Focal loss down-weights well-classified samples, focusing gradients on hard examples.

### Why Macro-F1 over Accuracy?
A model predicting "stroma" for everything achieves ~55% accuracy. Macro-F1 equally weights all classes, correctly exposing poor minority class performance.

## Current Status

- [x] **Session 1**: Data pipeline, CNN baseline, metric analysis
- [ ] **Session 2**: ViT with FlashAttention, distributed training
- [ ] **Session 3**: Multimodal VLM with LoRA
- [ ] **Session 4**: TensorRT INT8 optimization
- [ ] **Session 5**: Documentation, profiling, final polish

## License

This project is for educational and portfolio purposes.
