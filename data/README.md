# Data Directory

## Dataset Format

This project uses synthetic histopathology tile images for development and testing.

### Structure

```
data/
├── download_patches.py    # Script to generate synthetic dataset
├── manifest.csv           # Generated CSV with filepath, label, split columns
├── README.md              # This file
├── sample/                # 20 sample tiles for quick inspection
└── tiles/                 # All generated tiles (2,000 PNG files)
```

### Tile Specifications

- **Format**: PNG (lossless)
- **Size**: 1024 × 1024 pixels
- **Channels**: 3 (RGB)
- **Dtype**: uint8 (0–255)

### Classes

| Class    | Distribution | Visual Characteristics                       |
|----------|-------------|----------------------------------------------|
| tumor    | 15%         | Dense purple-stained nuclei on pink background |
| immune   | 20%         | Scattered small dark blue-purple nuclei       |
| stroma   | 55%         | Fibrous pink texture with wave-like patterns  |
| necrosis | 10%         | Pale, ghost-cell appearance, near-white       |

### Manifest CSV

The `manifest.csv` file contains:
- `filepath`: relative path to the tile PNG (from `data/`)
- `label`: one of `tumor`, `immune`, `stroma`, `necrosis`
- `split`: one of `train` (70%), `val` (15%), `test` (15%) — stratified by class

### Generating the Dataset

```bash
python data/download_patches.py --output_dir data/ --num_tiles 2000
```

### Why Synthetic Data?

This is a portfolio/research reproduction project. We use synthetic data because:
1. Real medical images require IRB approval and data use agreements
2. Synthetic data allows full reproducibility without access restrictions
3. The ML pipeline architecture is the focus, not the dataset itself
4. The synthetic tiles have class-specific visual features that make
   classification learnable, validating the pipeline works correctly
