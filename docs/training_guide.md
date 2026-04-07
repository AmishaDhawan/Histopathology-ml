# Training Guide

## Prerequisites

```bash
pip install -r requirements.txt
```

## Step 1: Generate Synthetic Dataset

```bash
python data/download_patches.py --output_dir data/ --num_tiles 2000
```

This creates 2,000 synthetic tiles in `data/tiles/` and a manifest at `data/manifest.csv`.

## Step 2: Train ResNet-50 Baseline

```bash
python scripts/train_resnet.py \
    --config configs/resnet_config.yaml \
    --data_dir data/ \
    --output_dir outputs/resnet/ \
    --device cuda
```

For CPU-only testing:
```bash
python scripts/train_resnet.py \
    --config configs/resnet_config.yaml \
    --data_dir data/ \
    --output_dir outputs/resnet/ \
    --device cpu \
    --epochs 2
```

### Outputs

- `outputs/resnet/best_model.pt` — best checkpoint (by validation macro-F1)
- `outputs/resnet/final_model.pt` — final epoch checkpoint
- `outputs/resnet/training_curves.png` — loss and metric curves
- `outputs/resnet/confusion_matrix.png` — test set confusion matrix

## Step 3: Explore Results

Open the Jupyter notebooks:
```bash
jupyter notebook notebooks/
```

- `01_data_exploration.ipynb` — dataset statistics and visualization
- `02_cnn_baseline.ipynb` — training analysis and metric discussion
