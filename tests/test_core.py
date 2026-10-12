import csv
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
import yaml

from crconv import CRConv2d, DBSN, beta_schedule, revisible_loss
from crconv.data import ImageDataset, load_gray

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


class CoreTest(unittest.TestCase):
    def test_fmd_manifest_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for kind in ('A', 'B'):
                for scene in (1, 2):
                    raw = directory / 'dataset' / kind / 'raw' / str(scene)
                    raw.mkdir(parents=True)
                    for index in (0, 1):
                        Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(raw / f'{index}.png')
                gt = directory / 'dataset' / kind / 'gt' / '2'
                gt.mkdir(parents=True)
                Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(gt / 'reference.png')
            result = subprocess.run([sys.executable, 'scripts/make_fmd_manifests.py',
                                     '--root', str(directory / 'dataset'),
                                     '--output', str(directory / 'manifests'),
                                     '--train-scenes', '1', '--val-scenes', '2', '--val-limit', '2'],
                                    cwd=ROOT, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            training = ImageDataset(directory / 'manifests/train.csv', training=True)
            validation = ImageDataset(directory / 'manifests/val.csv')
            self.assertEqual(len(training), 4)
            self.assertEqual(len(validation), 2)
            self.assertEqual({Path(r['noisy']).parts[-4] for r in validation.samples}, {'A', 'B'})
            self.assertEqual(validation[0][0].shape, (1, 8, 8))

    def test_completion(self):
        for rule in ('kernel_softmax', 'tied_mean', 'harmonic_4'):
            layer = CRConv2d(2, 3, 3, padding=1, center_mode=rule).double()
            w = layer.weight.detach().clone().flatten(-2)
            neighbors = torch.cat((w[..., :4], w[..., 5:]), -1)
            center = layer.effective_weight(False)[..., 1, 1]
            self.assertTrue(torch.all(center >= neighbors.min(-1).values - 1e-12))
            self.assertTrue(torch.all(center <= neighbors.max(-1).values + 1e-12))
            if rule == 'tied_mean':
                torch.testing.assert_close(center, neighbors.mean(-1))
            before = layer.effective_weight(False).detach().clone()
            with torch.no_grad():
                layer.weight[..., 1, 1] = 1000
            torch.testing.assert_close(before, layer.effective_weight(False))
            with torch.no_grad():
                layer.weight.zero_()
            self.assertTrue(torch.isfinite(layer.effective_weight(False)).all())

    def test_blind_safety_including_boundaries(self):
        torch.manual_seed(7)
        model = DBSN(base_channels=4, blocks=2).double()
        x = torch.randn(1, 1, 13, 13, dtype=torch.float64, requires_grad=True)
        for y, z in [(0, 0), (0, 6), (6, 6), (12, 12)]:
            gradient = torch.autograd.grad(model(x, blind=True)[0, 0, y, z], x)[0]
            self.assertEqual(gradient[0, 0, y, z].item(), 0)
        layer = CRConv2d(1, 1, 3, padding=1).double()
        with torch.no_grad():
            layer.weight.fill_(1)
        gradient = torch.autograd.grad(layer(x, blind=False)[0, 0, 6, 6], x)[0]
        self.assertGreater(gradient[0, 0, 6, 6].item(), 0)

    def test_loss_gradient_and_schedule(self):
        b = torch.randn(2, 1, 9, 9, requires_grad=True)
        v = torch.randn_like(b, requires_grad=True)
        y = torch.randn_like(b)
        loss = revisible_loss(b, v, y, alpha=1, beta=3)
        loss.backward()
        self.assertIsNone(v.grad)
        expected = 2 * ((b.detach() - y) + (b.detach() - y + 3 * (v.detach() - y))) / b.numel()
        torch.testing.assert_close(b.grad, expected)
        self.assertEqual(beta_schedule(40, 100), 2)
        self.assertAlmostEqual(beta_schedule(41, 100), 2.3)
        self.assertEqual(beta_schedule(100, 100), 20)

    def test_data_and_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            rng = np.random.default_rng(4)
            for i in range(4):
                Image.fromarray(rng.integers(0, 256, (16, 16), dtype=np.uint8)).save(directory / f'{i}.png')
            Image.fromarray(np.array([[0, 65535]], dtype=np.uint16)).save(directory / 'u16.png')
            np.testing.assert_equal(load_gray(directory / 'u16.png'), [[0., 1.]])
            for name, rows in [('train', [('0.png', ''), ('1.png', '')]),
                               ('val', [('2.png', '3.png')])]:
                with (directory / f'{name}.csv').open('w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['noisy', 'clean'])
                    writer.writerows(rows)
            dataset = ImageDataset(directory / 'train.csv', training=True, patch_size=8)
            self.assertEqual(dataset[0].shape, (1, 8, 8))
            config = dict(model=dict(base_channels=4, blocks=1),
                          data=dict(train_csv='train.csv', val_csv='val.csv', patch_size=8,
                                    batch_size=2, repeat=1, workers=0),
                          training=dict(epochs=2, lr=.0001, seed=42, objective='revisible',
                                        alpha=1, beta_start=2, beta_end=20, warm_fraction=.4),
                          output='run')
            path = directory / 'config.yaml'
            path.write_text(yaml.safe_dump(config))
            env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
            def run(*args):
                result = subprocess.run([sys.executable, *args], cwd=ROOT, env=env,
                                        capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            interrupt = ('import train\noriginal = train.evaluate\ncalls = 0\n'
                         'def evaluate(*args, **kwargs):\n'
                         ' global calls\n calls += 1\n'
                         ' if calls == 2: raise SystemExit(0)\n'
                         ' return original(*args, **kwargs)\n'
                         'train.evaluate = evaluate\ntrain.main()\n')
            run('-c', interrupt, '--config', str(path), '--device', 'cpu')
            checkpoint = directory / 'run/latest.pth'
            state = torch.load(checkpoint, weights_only=False)
            self.assertEqual(state['epoch'], 1)
            run('train.py', '--config', str(path), '--device', 'cpu', '--resume')
            self.assertEqual(torch.load(checkpoint, weights_only=False)['epoch'], 2)
            run('train.py', '--config', str(path), '--device', 'cpu', '--output', str(directory / 'continuous'))
            resumed = torch.load(checkpoint, weights_only=False)
            continuous = torch.load(directory / 'continuous/latest.pth', weights_only=False)
            for key in resumed['model']:
                torch.testing.assert_close(resumed['model'][key], continuous['model'][key], rtol=0, atol=0)
            run('infer.py', '--checkpoint', str(directory / 'run/best.pth'),
                '--manifest', str(directory / 'val.csv'), '--output', str(directory / 'predictions'), '--device', 'cpu')
            self.assertTrue((directory / 'predictions/metrics.csv').is_file())
            self.assertEqual(np.load(directory / 'predictions/0000_2.npy').shape, (16, 16))
            run('train.py', '--config', str(path), '--device', 'cpu',
                '--objective', 'blind', '--output', str(directory / 'blind'))


if __name__ == '__main__':
    unittest.main()
