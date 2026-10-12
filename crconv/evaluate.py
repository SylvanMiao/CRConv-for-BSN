import numpy as np
from pathlib import Path
from PIL import Image
from skimage.metrics import structural_similarity
import torch


@torch.no_grad()
def evaluate(model, loader, device, blind, output=None):
    model.eval()
    records = []
    for noisy, clean, paths in loader:
        prediction = model(noisy.to(device), blind=blind).clamp(0, 1).cpu().numpy()
        for pred, target, path in zip(prediction[:, 0], clean.numpy()[:, 0], paths):
            mse = np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2)
            score = float(-10 * np.log10(max(mse, 1e-12)))
            similarity = float(structural_similarity(target, pred, data_range=1.0))
            records.append({'image': path, 'psnr': score, 'ssim': similarity})
            if output is not None:
                name = f'{len(records) - 1:04d}_{Path(path).stem}'
                np.save(output / f'{name}.npy', pred)
                Image.fromarray(np.rint(pred * 65535).astype(np.uint16)).save(output / f'{name}.png')
    return records
