# Architecture Decisions

## Overview

This project implements a four-phase ML pipeline for histopathology image analysis:

1. **Phase 1: CNN Baseline** — ResNet-50 fine-tuned for tile classification
2. **Phase 2: Vision Transformer** — ViT-Base with FlashAttention and distributed training
3. **Phase 3: Multimodal VLM** — LLaVA-style vision-language model with LoRA
4. **Phase 4: Production Inference** — TensorRT INT8 optimization

## Phase 1: CNN Baseline

### Why ResNet-50?

ResNet-50 is the standard baseline for medical image classification because:
- ImageNet pretrained features (edges, textures, color gradients) transfer well
- Well-understood architecture — easy to debug and benchmark
- Fast to train — establishes a performance floor in hours, not days
- 25.6M parameters — sufficient capacity for 4-class classification

### Why fine-tune ALL layers?

Histopathology features differ significantly from ImageNet:
- Low-level features (edges) transfer, but mid/high-level features (organ structures, cell morphology) require adaptation
- Empirically, full fine-tuning outperforms frozen-backbone approaches on pathology datasets
- With only ~2000 samples, dropout (0.3) prevents overfitting

### Why Focal Loss?

Class distribution: stroma (55%), immune (20%), tumor (15%), necrosis (10%)
- Standard cross-entropy is dominated by the majority class
- Focal loss down-weights easy, well-classified examples
- gamma=2.0 is the standard choice for moderate imbalance (5.5:1 ratio)

### Why Macro-F1 over Accuracy?

A model predicting "stroma" for everything achieves ~55% accuracy.
Macro-F1 equally weights all classes, exposing poor minority class performance.
