# Technical Debts & Development Log

Comprehensive record of technical hurdles, hardware limitations, and architectural decisions
across all sessions of the Histopathology ML project.

---

## Session 1 — CNN Baseline (ResNet-50)

### Roadblocks Encountered

#### 1. Synthetic Data Generation with Perlin Noise
- **Issue**: The `noise` library (`pnoise2`) required careful parameter tuning to produce
  tissue-like synthetic textures. Initial attempts produced either uniform or excessively noisy
  images that didn't resemble histopathology tiles.
- **Resolution**: Tuned octaves, persistence, and lacunarity per class (tumor, immune, stroma,
  necrosis) to produce visually distinct synthetic patterns. Added class-specific color palettes
  to mimic H&E staining.

#### 2. Class Imbalance (55% Stroma)
- **Issue**: Accuracy as a metric was misleading — a model predicting "stroma" for every tile
  achieves 55% accuracy. Early training runs appeared to converge but were actually learning
  the majority class.
- **Resolution**: Switched primary metric to **macro-F1**. Added `WeightedRandomSampler` to
  oversample minority classes (tumor=15%, necrosis=10%). Implemented `FocalLoss(gamma=2.0)`
  to down-weight easy/majority examples.

#### 3. CI Pipeline Configuration
- **Issue**: flake8 and pytest needed specific configuration to work with the project structure
  (e.g., `sys.path.insert` in scripts, `E402` import order violations).
- **Resolution**: Configured `.github/workflows/ci.yml` with
  `flake8 src/ scripts/ tests/ --max-line-length=120 --ignore=E402,W503,E501`.

---

## Session 2 — ViT + FlashAttention + Distributed Training

### Roadblocks Encountered

#### 1. CPU Bottleneck — ViT-Base Training Infeasible on CPU
- **Severity**: Critical (blocks full checkpoint verification)
- **Issue**: ViT-Base has 85.8M parameters with 12 transformer blocks, each performing
  multi-head self-attention over 197 tokens (196 patches + 1 CLS token). On CPU, a single
  forward+backward pass for a batch of 16 takes approximately 10 minutes. A full 2-epoch
  training run (88 batches x 2 = 176 steps) would take ~29 hours.
- **Root cause**: Self-attention is O(n^2) in sequence length and requires dense matrix
  multiplications. CPUs lack the massive parallelism of GPU tensor cores needed for efficient
  matrix operations. FlashAttention (`F.scaled_dot_product_attention`) provides no speedup
  on CPU — it only selects optimized CUDA kernels on GPU.
- **Impact**: Training script verified to correctly load pretrained weights and begin training,
  but no epoch-level metrics (loss, F1, AUROC) were produced.
- **Workaround**: Verified model correctness through unit tests (forward pass shape, attention
  map extraction, parameter counting). Full training requires `--device cuda`.
- **Status**: Open — requires GPU access for end-to-end verification.

#### 2. Pretrained Weight Loading Crash (`MLPBlock.linear_1` AttributeError)
- **Severity**: Critical (crash on model initialization)
- **Issue**: Loading pretrained ViT-Base/16 weights from `torchvision.models.vit_b_16()` into
  our custom `ViTHistoClassifier` crashed with:
  ```
  AttributeError: 'MLPBlock' object has no attribute 'linear_1'
  ```
- **Root cause**: In torchvision >= 0.14, `MLPBlock` was refactored to inherit from
  `nn.Sequential` instead of using named attributes. The MLP layers are accessed by index:
  - `mlp[0]` = first `nn.Linear` (768 -> 3072)
  - `mlp[1]` = `nn.Dropout`
  - `mlp[2]` = `nn.GELU`
  - `mlp[3]` = second `nn.Linear` (3072 -> 768)
  - `mlp[4]` = `nn.Dropout`
- **Fix**: Changed weight copying from:
  ```python
  src_block.mlp.linear_1.weight  # AttributeError
  src_block.mlp.linear_2.weight  # AttributeError
  ```
  to:
  ```python
  src_block.mlp[0].weight  # First Linear layer
  src_block.mlp[3].weight  # Second Linear layer
  ```
- **Lesson**: Always inspect the actual module structure with `print(module)` or
  `list(module.named_children())` rather than relying on documentation or naming conventions.
- **Status**: Fixed (commit `ead1a12`).

#### 3. CosineAnnealingLR with Negative T_max
- **Severity**: Medium (crash with specific CLI arguments)
- **Issue**: Running `--epochs 2` with `warmup_epochs: 10` in config produced
  `T_max = 2 - 10 = -8`, which would cause `CosineAnnealingLR` to behave unpredictably
  or crash.
- **Root cause**: No guard on scheduler creation — assumed `num_epochs > warmup_epochs`.
- **Fix**: Added guard:
  ```python
  if config["training"]["lr_scheduler"] == "cosine" and num_epochs > warmup_epochs:
      scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)
  else:
      scheduler = None
  ```
- **Lesson**: Always validate derived parameters at construction time, especially when
  CLI arguments can override config values.
- **Status**: Fixed (commit `ead1a12`).

#### 4. `get_attention_maps()` Leaving Model in Eval Mode
- **Severity**: Medium (silent correctness bug)
- **Issue**: `get_attention_maps()` called `self.eval()` to disable dropout during attention
  extraction but never restored the model to training mode. If called mid-training, all
  subsequent forward passes would run without dropout, silently degrading regularization.
- **Root cause**: Asymmetry with the existing `use_flash` save/restore pattern — the flash
  attention flag was correctly restored, but training mode was not.
- **Fix**:
  ```python
  was_training = self.training
  self.eval()
  # ... extract attention maps ...
  if was_training:
      self.train()
  ```
- **Lesson**: Any method that changes model state (`eval()`, `train()`, disabling modules)
  must save and restore the original state. Use context managers when possible.
- **Status**: Fixed (commit `e36fc70`).

#### 5. Adam Optimizer Memory Overestimate (3x vs 2x)
- **Severity**: Low (incorrect documentation/estimates, no runtime impact)
- **Issue**: `compute_memory_footprint()` in `DistributedTrainer` estimated Adam optimizer
  memory as `3 * param_mb` with a comment claiming Adam stores "parameters + first moment +
  second moment = 3x". This overestimates by 50%.
- **Root cause**: Standard AdamW stores only `exp_avg` (first moment) and `exp_avg_sq`
  (second moment) — 2x parameter size. The optimizer holds *references* to model parameters,
  not copies. The "3x" figure applies only to implementations that maintain separate
  parameter copies (e.g., some mixed-precision training setups).
- **Fix**: Changed to `optimizer_mb = 2 * param_mb` and updated all comments and docstrings.
- **Impact on estimates**:
  - DDP memory: corrected from 5x to 4x param size (params + grads + 2x optimizer)
  - FSDP memory: corrected proportionally
- **Status**: Fixed (commit `e36fc70`).

#### 6. Training Log Header Misalignment
- **Severity**: Low (cosmetic)
- **Issue**: Table header defined 7 columns but data rows printed 8 values (extra elapsed
  time column), causing misaligned output.
- **Fix**: Added `| {'Time':>6}` to header format string.
- **Status**: Fixed (commit `cfd9a72`).

---

## Known Technical Debts

### High Priority

1. **No GPU-verified training metrics**: ViT-Base training has not been run to completion.
   The notebook contains projected/aspirational metrics (macro-F1=0.84) but no measured values.
   Must run on GPU to produce real training curves, confusion matrices, and attention maps.

2. **Pretrained weight source**: Currently using supervised ImageNet weights
   (`ViT_B_16_Weights.IMAGENET1K_V1`). The architecture docstring recommends DINO
   self-supervised weights, which produce better features for pathology downstream tasks
   (per CONCH/UNI papers). Switching to DINO weights requires loading from a different
   checkpoint format.

3. **No mixed-precision training**: The training script runs in full fp32. Adding
   `torch.cuda.amp.autocast()` with `GradScaler` would approximately halve memory usage
   and significantly improve GPU training throughput.

### Medium Priority

4. **Distributed training untested**: `DistributedTrainer` class implements DDP and FSDP
   patterns but has not been tested on multi-GPU hardware. The `--distributed` flag in
   `train_vit.py` sets up the process group but the full training loop with gradient
   synchronization is not end-to-end verified.

5. **No CPU memory tracking**: `log_memory()` only tracks GPU memory via
   `torch.cuda.memory_allocated()`. CPU training runs produce no memory diagnostics.
   **Resolution**: Added `psutil`-based RSS logging (see below).

6. **Hardcoded ViT-Base architecture**: The `ViTHistoClassifier` is hardcoded to ViT-Base
   dimensions (embed_dim=768, num_heads=12, depth=12). Supporting ViT-Small or ViT-Large
   would require parameterizing these values.

### Low Priority

7. **Notebook simulated metrics**: `03_vit_training.ipynb` generates synthetic training
   curves and confusion matrices for narrative purposes. These should be replaced with
   real training outputs once GPU training completes.

8. **Attention map resolution**: Attention maps are extracted at 14x14 resolution
   (224/16 = 14 patches per side) and upsampled with bilinear interpolation. For clinical
   use, higher-resolution attention (patch_size=8 or overlapping patches) would be needed.

9. **No learning rate finder**: The learning rate (1e-4) is set based on literature
   conventions. A learning rate range test (Smith 2017) could find a better value for
   this specific dataset/model combination.

---

## Future Development Roadmap

### Session 3 — Multimodal Vision-Language Model
- **Goal**: Combine ViT visual features with text descriptions using a projection layer
  and language model integration.
- **Key files**: `src/models/projection_layer.py`, `src/models/multimodal_vlm.py`,
  `scripts/train_vlm.py`, `notebooks/04_multimodal_llm.ipynb`
- **Prerequisites**: GPU-verified ViT checkpoint from Session 2 (or use pretrained weights
  directly).
- **Expected challenges**: Tokenizer integration, cross-modal attention alignment, managing
  memory for combined vision+language model.

### Session 4 — TensorRT Inference Optimization
- **Goal**: Export trained models to TensorRT for optimized inference, implement batched
  inference pipeline.
- **Key files**: `scripts/export_trt.py`, `scripts/serve.py`,
  `notebooks/05_inference_optimization.ipynb`
- **Prerequisites**: Working PyTorch model checkpoint, NVIDIA GPU with TensorRT installed.
- **Expected challenges**: Dynamic shape handling, operator support in TensorRT, quantization
  accuracy trade-offs (INT8 calibration).

### Session 5 — Profiling & Performance Analysis
- **Goal**: Full performance profiling with Nsight Systems, roofline analysis, and
  optimization recommendations.
- **Key files**: `src/profiling/pytorch_profiler.py`, `src/profiling/roofline.py`
- **Prerequisites**: GPU training runs with NVTX annotations enabled.
- **Expected challenges**: Interpreting profiler traces, identifying actual bottlenecks
  vs. profiler overhead.

### Cross-Session Improvements
- **GPU migration**: All training scripts should be validated on CUDA hardware. Consider
  adding a CI job with GPU runners (GitHub Actions with NVIDIA GPU, or self-hosted runner).
- **Data augmentation expansion**: Current augmentations are conservative (90-degree rotations,
  mild color jitter). Consider adding Cutout, MixUp, or stain normalization transforms
  specific to histopathology.
- **Experiment tracking**: Integrate Weights & Biases or MLflow for proper experiment
  tracking, hyperparameter logging, and artifact management. Replace manual metric tracking
  with a proper experiment framework.
- **Model registry**: Implement a model registry for versioned checkpoints with metadata
  (training config, metrics, git commit hash).
- **Real data pipeline**: The current pipeline uses synthetic data. When real histopathology
  tiles are available, the data loading and augmentation pipeline should be validated with
  actual H&E-stained tissue images.
