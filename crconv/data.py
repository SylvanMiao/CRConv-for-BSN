import csv
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def load_gray(path, intensity_max=None):
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(f"Expected a single-channel image, got {array.shape}: {path}")
    if intensity_max is None:
        if array.dtype == np.uint8:
            intensity_max = 255
        elif array.dtype.kind == 'u' and array.dtype.itemsize == 2:
            intensity_max = 65535
        elif array.dtype.kind == 'f':
            intensity_max = 1
        else:
            raise ValueError(f"Set intensity_max explicitly for {array.dtype}: {path}")
    if not np.isfinite(intensity_max) or intensity_max <= 0:
        raise ValueError("intensity_max must be positive and finite")
    array = array.astype(np.float32) / intensity_max
    if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
        raise ValueError(f"Image must normalize to [0,1]: {path}")
    return array


class ImageDataset(Dataset):
    """CSV paths resolve relative to the CSV file; clean targets are evaluation-only."""

    def __init__(self, manifest, training=False, patch_size=None, repeat=1, intensity_max=None):
        self.manifest = Path(manifest).resolve()
        with self.manifest.open(newline='') as handle:
            self.samples = list(csv.DictReader(handle))
        if not self.samples or 'noisy' not in self.samples[0]:
            raise ValueError("Manifest must contain a noisy column and at least one row")
        self.training, self.patch_size = training, patch_size
        self.repeat = repeat if training else 1
        self.intensity_max = intensity_max
        if self.repeat < 1 or (patch_size is not None and patch_size < 1):
            raise ValueError("repeat and patch_size must be positive")
        for row in self.samples:
            for key in ('noisy',) if training else ('noisy', 'clean'):
                if not row.get(key):
                    raise ValueError(f"Missing {key} path")
                path = (self.manifest.parent / row[key]).resolve()
                if not path.is_file():
                    raise FileNotFoundError(path)
                row[key] = str(path)

    def __len__(self):
        return len(self.samples) * self.repeat

    def __getitem__(self, index):
        row = self.samples[index % len(self.samples)]
        noisy = load_gray(row['noisy'], self.intensity_max)
        clean = None if self.training else load_gray(row['clean'], self.intensity_max)
        if clean is not None and clean.shape != noisy.shape:
            raise ValueError(f"Reference shape mismatch: {row['noisy']}")
        if self.patch_size:
            h, w = noisy.shape
            p = self.patch_size
            if min(h, w) < p:
                raise ValueError(f"Image smaller than crop: {row['noisy']}")
            if self.training:
                top = int(torch.randint(h - p + 1, ()).item())
                left = int(torch.randint(w - p + 1, ()).item())
            else:
                top, left = (h - p) // 2, (w - p) // 2
            noisy = noisy[top:top + p, left:left + p]
            if clean is not None:
                clean = clean[top:top + p, left:left + p]
        if self.training:
            if torch.rand(()) > .5:
                noisy = np.flip(noisy, 0)
            if torch.rand(()) > .5:
                noisy = np.flip(noisy, 1)
            noisy = np.rot90(noisy, int(torch.randint(4, ()).item()))
        tensor = torch.from_numpy(np.ascontiguousarray(noisy)).unsqueeze(0)
        if self.training:
            return tensor
        return tensor, torch.from_numpy(np.ascontiguousarray(clean)).unsqueeze(0), row['noisy']
