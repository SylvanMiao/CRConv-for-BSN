# CRConv for Blind-Spot Networks

A minimal PyTorch example with CRConv, a no-PD D-BSN backbone, and re-visible training.

## Install

```bash
python -m pip install -r requirements.txt
```

## Prepare data

Use grayscale images. Create `data/train.csv`:

```csv
noisy
/path/to/train/image001.png
/path/to/train/image002.png
```

Create `data/val.csv` and a separate `data/test.csv`:

```csv
noisy,clean
/path/to/raw/image001.png,/path/to/reference/image001.png
```

Clean references are used only for validation and evaluation. Keep training, validation, and test scenes separate. CSV paths are relative to the CSV file; config paths are relative to the YAML file.

For FMD organized as `ROOT/<type>/raw/<scene>/*` and `ROOT/<type>/gt/<scene>/*`, generate training/validation manifests with:

```bash
python scripts/make_fmd_manifests.py --root /path/to/FMD_ROOT --output data
```

Defaults: training scenes 1–18, validation scenes 19–20, 10 validation images. Prepare the test CSV separately.

## Train

Edit `configs/dbsn_crconv.yaml` for data paths, batch size, and training settings. Images are normalized by 255 (8-bit) or 65535 (16-bit); floating-point images must be in [0,1]. Set `data.intensity_max` explicitly for other intensity ranges.

CRConv:

```bash
CUDA_VISIBLE_DEVICES=0 python -u train.py --config configs/dbsn_crconv.yaml
```

Blind baseline:

```bash
CUDA_VISIBLE_DEVICES=1 python -u train.py --config configs/dbsn_crconv.yaml \
  --objective blind --output runs/dbsn_blind_seed42
```

Resume CRConv training from the last completed epoch:

```bash
CUDA_VISIBLE_DEVICES=0 python -u train.py --config configs/dbsn_crconv.yaml --resume
```

Logs, `history.csv`, `latest.pth`, and validation-selected `best.pth` are saved under the configured output directory.

## Evaluate

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py \
  --checkpoint runs/dbsn_crconv_seed42/best.pth \
  --manifest data/test.csv --output runs/dbsn_crconv_test
```

Outputs: PSNR/SSIM in `metrics.csv`, floating-point `.npy` predictions, and 16-bit PNGs. Inference uses the visible state for CRConv and the blind state for the baseline. Only load trusted checkpoints.

## Optional tests

```bash
python -m unittest discover -s tests -v
```

Tests run on CPU with synthetic data and cover center completion, blind-spot gradients, data loading, training, resume, and inference.
