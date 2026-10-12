import argparse
import csv
import os
from pathlib import Path

EXTENSIONS = {'.png', '.tif', '.tiff', '.bmp', '.jpg', '.jpeg'}


def images(path):
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS) if path.is_dir() else []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('data'))
    parser.add_argument('--train-scenes', type=int, nargs='+', default=list(range(1, 19)))
    parser.add_argument('--val-scenes', type=int, nargs='+', default=[19, 20])
    parser.add_argument('--val-limit', type=int, default=10)
    args = parser.parse_args()
    if set(args.train_scenes) & set(args.val_scenes):
        raise ValueError('Scene splits overlap')
    if args.val_limit < 0:
        raise ValueError('val-limit must be >= 0 (0 means all)')
    root, output = args.root.resolve(), args.output.resolve()
    training, groups = [], []
    kinds = sorted(p for p in root.iterdir() if (p / 'raw').is_dir())
    for scene in sorted(set(args.train_scenes + args.val_scenes)):
        for kind in kinds:
            raw = images(kind / 'raw' / str(scene))
            if not raw:
                raise FileNotFoundError(kind / 'raw' / str(scene))
            if scene in args.train_scenes:
                training.extend({'noisy': os.path.relpath(p, output)} for p in raw)
            else:
                references = images(kind / 'gt' / str(scene))
                group = []
                for p in raw:
                    matches = [q for q in references if q.name == p.name]
                    if not matches:
                        matches = [q for q in references if q.stem == p.stem]
                    if not matches and len(references) == 1:
                        matches = references
                    if len(matches) != 1:
                        raise ValueError(f'Ambiguous or missing reference for {p}')
                    group.append(dict(noisy=os.path.relpath(p, output),
                                      clean=os.path.relpath(matches[0], output)))
                groups.append(group)
    validation = [group[i] for i in range(max(map(len, groups), default=0))
                  for group in groups if i < len(group)]
    if args.val_limit:
        validation = validation[:args.val_limit]
    if not training or not validation:
        raise ValueError('Empty split')
    if any((output / f'{name}.csv').exists() for name in ('train', 'val')):
        raise FileExistsError('Refusing to overwrite existing manifests')
    output.mkdir(parents=True, exist_ok=True)
    for name, rows, fields in [('train', training, ['noisy']), ('val', validation, ['noisy', 'clean'])]:
        with (output / f'{name}.csv').open('x', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f'{name}: {len(rows)} images -> {output / (name + ".csv")}')


if __name__ == '__main__':
    main()
