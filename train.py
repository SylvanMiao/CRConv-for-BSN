import argparse
import csv
import logging
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from crconv import DBSN, beta_schedule, revisible_loss
from crconv.data import ImageDataset
from crconv.evaluate import evaluate


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def atomic_save(state, path):
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/dbsn_crconv.yaml')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--objective', choices=['blind', 'revisible'])
    parser.add_argument('--output')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    path = Path(args.config).resolve()
    config = yaml.safe_load(path.read_text())
    for key in ('train_csv', 'val_csv'):
        config['data'][key] = str((path.parent / config['data'][key]).resolve())
    config['output'] = str(Path(args.output).resolve() if args.output
                           else (path.parent / config['output']).resolve())
    if args.objective:
        config['training']['objective'] = args.objective
    tc, dc = config['training'], config['data']
    if tc['objective'] not in {'blind', 'revisible'}:
        raise ValueError('objective must be blind or revisible')
    beta_schedule(1, tc['epochs'], tc['beta_start'], tc['beta_end'], tc['warm_fraction'])
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu for a smoke test')
    random.seed(tc['seed'])
    np.random.seed(tc['seed'])
    torch.manual_seed(tc['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    generator = torch.Generator().manual_seed(tc['seed'])
    training = ImageDataset(dc['train_csv'], True, dc['patch_size'], dc['repeat'], dc.get('intensity_max'))
    validation = ImageDataset(dc['val_csv'], False, dc['patch_size'], intensity_max=dc.get('intensity_max'))
    if {r['noisy'] for r in training.samples} & {r['noisy'] for r in validation.samples}:
        raise ValueError('Training and validation manifests overlap')
    train_loader = DataLoader(training, batch_size=dc['batch_size'], shuffle=True,
                              num_workers=dc['workers'], drop_last=True,
                              worker_init_fn=seed_worker, generator=generator)
    val_loader = DataLoader(validation, batch_size=1, num_workers=0)
    if not len(train_loader):
        raise ValueError('Training set is smaller than one batch')
    model = DBSN(**config['model']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tc['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tc['epochs'])
    output = Path(config['output'])
    history, start, best = [], 1, -float('inf')
    if args.resume:
        state = torch.load(output / 'latest.pth', map_location='cpu', weights_only=False)
        if state['config'] != config:
            raise ValueError('Resume configuration differs from checkpoint')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        generator.set_state(state['generator_rng'])
        torch.set_rng_state(state['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        history, start, best = state['history'], state['epoch'] + 1, state['best_psnr']
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(output / 'train.log')])
    logging.info('objective=%s parameters=%d batches=%d validation_images=%d device=%s',
                 tc['objective'], sum(p.numel() for p in model.parameters()),
                 len(train_loader), len(validation), device)
    for epoch in range(start, tc['epochs'] + 1):
        model.train()
        beta = beta_schedule(epoch, tc['epochs'], tc['beta_start'], tc['beta_end'], tc['warm_fraction'])
        total = 0.0
        for noisy in tqdm(train_loader, desc=f'{epoch}/{tc["epochs"]}'):
            noisy = noisy.to(device)
            optimizer.zero_grad(set_to_none=True)
            if tc['objective'] == 'revisible':
                with torch.no_grad():
                    visible = model(noisy, blind=False)
                blind = model(noisy, blind=True)
                loss = revisible_loss(blind, visible, noisy, tc['alpha'], beta)
            else:
                loss = (model(noisy, blind=True) - noisy).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            loss.backward()
            optimizer.step()
            total += loss.item()
        results = evaluate(model, val_loader, device, blind=tc['objective'] == 'blind')
        psnr = float(np.mean([r['psnr'] for r in results]))
        ssim = float(np.mean([r['ssim'] for r in results]))
        improved = psnr > best
        best = max(best, psnr)
        scheduler.step()
        history.append(dict(epoch=epoch, beta=beta, loss=total / len(train_loader),
                            val_psnr=psnr, val_ssim=ssim, best_psnr=best,
                            lr=scheduler.get_last_lr()[0]))
        state = dict(epoch=epoch, config=config, model=model.state_dict(),
                     optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                     best_psnr=best, history=history, generator_rng=generator.get_state(),
                     torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [])
        if improved:
            atomic_save(state, output / 'best.pth')
        atomic_save(state, output / 'latest.pth')
        with (output / 'history.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        logging.info('epoch=%d beta=%.3f loss=%.7f val=%.4f/%.5f best=%.4f',
                     epoch, beta, total / len(train_loader), psnr, ssim, best)


if __name__ == '__main__':
    main()
