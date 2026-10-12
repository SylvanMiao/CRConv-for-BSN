import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from crconv import DBSN
from crconv.data import ImageDataset
from crconv.evaluate import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--mode', choices=['blind', 'visible'])
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = state['config']
    model = DBSN(**config['model']).to(args.device)
    model.load_state_dict(state['model'])
    model.eval()
    mode = args.mode or ('blind' if config['training']['objective'] == 'blind' else 'visible')
    dataset = ImageDataset(args.manifest, intensity_max=config['data'].get('intensity_max'))
    loader = DataLoader(dataset, batch_size=1)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    records = evaluate(model, loader, args.device, blind=mode == 'blind', output=output)
    records.append(dict(image='AVERAGE', psnr=float(np.mean([r['psnr'] for r in records])),
                        ssim=float(np.mean([r['ssim'] for r in records]))))
    with (output / 'metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['image', 'psnr', 'ssim'])
        writer.writeheader()
        writer.writerows(records)
    print(f'epoch={state["epoch"]} mode={mode} {records[-1]}', flush=True)


if __name__ == '__main__':
    main()
