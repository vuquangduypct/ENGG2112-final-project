"""Deterministic image preparation. Originals are never overwritten."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw
from .common import ROOT, file_hash, save_json


def prepare_image(image, size=224, crop_black=False, threshold=7):
    rgb = image.convert('RGB')
    bounds = (0, 0, rgb.width, rgb.height)
    if crop_black:
        # Bounding box of non-dark pixels; optional experiment, not a clinically validated segmentation.
        mask = Image.fromarray((np.asarray(rgb).max(axis=2) > threshold).astype('uint8') * 255)
        box = mask.getbbox()
        if box is not None:
            bounds = box
            rgb = rgb.crop(box)
    # Preserve aspect ratio; do not stretch a circular retina into an ellipse.
    prepared = ImageOps.pad(rgb, (size, size), method=Image.Resampling.LANCZOS, color=(0, 0, 0))
    return prepared, bounds


def run_preprocess(manifest_path, image_dir, output_dir, size=224, crop_black=False,
                   threshold=7, workers=4, limit=None):
    if size < 16 or not 0 <= threshold <= 255:
        raise ValueError('Invalid image size or black-border threshold.')
    if output_dir.resolve() == image_dir.resolve() or image_dir.resolve() in output_dir.resolve().parents:
        raise ValueError('Output must be outside the original image directory.')
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError('Output already contains files. Choose a new versioned directory.')
    frame = pd.read_csv(manifest_path, dtype={'id_code': str})
    if not {'id_code', 'diagnosis', 'split'} <= set(frame.columns):
        raise ValueError('Manifest is missing required columns.')
    if not frame.id_code.str.fullmatch(r'[A-Za-z0-9_-]+').all() or frame.id_code.duplicated().any():
        raise ValueError('Unsafe or duplicate ID in manifest.')
    if not frame.diagnosis.isin(range(5)).all() or not frame.split.isin(['train', 'validation', 'test']).all():
        raise ValueError('Unexpected grade or split.')
    if limit is not None:
        if limit < 1:
            raise ValueError('Limit must be positive.')
        frame = frame.head(limit)
    output_dir.mkdir(parents=True, exist_ok=True)
    def convert(row):
        path = image_dir / f'{row.id_code}.png'
        with Image.open(path) as source:
            prepared, box = prepare_image(source, size, crop_black, threshold)
        target = output_dir / f'{row.id_code}.png'
        prepared.save(target)
        return {'id_code': row.id_code, 'diagnosis': int(row.diagnosis), 'split': row.split,
                'relative_path': target.name, 'crop_box': ','.join(map(str, box))}
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for number, item in enumerate(pool.map(convert, frame.itertuples()), 1):
            results.append(item)
            if number % 500 == 0 or number == len(frame):
                print(f'Prepared {number}/{len(frame)} images', flush=True)
    pd.DataFrame(results).to_csv(output_dir / 'processed_manifest.csv', index=False)
    config = {'status': 'EXPERIMENTAL BASELINE - confirm model input requirements with teammate',
              'size': size, 'mode': 'RGB', 'format': 'lossless PNG, uint8 0..255',
              'resize': 'aspect-preserving LANCZOS resize and black padding',
              'crop_black': crop_black, 'black_threshold': threshold, 'rows': len(results),
              'manifest_sha256': file_hash(manifest_path), 'augmentation': 'none',
              'normalization': 'none; apply model-specific normalization in training/inference code',
              'notes': ['224 is a compute-friendly baseline, not a validated optimal size for retinal lesions.',
                        'No clinical claim about preservation of small lesions; review full-resolution examples.',
                        'Use the same deterministic transform for train, validation and test.']}
    save_json(output_dir / 'preprocessing_config.json', config)
    # One local before/after example for each available grade.
    sample_rows = frame.groupby('diagnosis').head(1)
    sheet = Image.new('RGB', (440, len(sample_rows) * 250), 'white')
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(sample_rows.itertuples()):
        for col, source_path in enumerate([image_dir / f'{row.id_code}.png', output_dir / f'{row.id_code}.png']):
            with Image.open(source_path) as source:
                preview = ImageOps.pad(source.convert('RGB'), (210, 210))
            sheet.paste(preview, (col * 220, index * 250))
        draw.text((5, index * 250 + 215), f'Grade {row.diagnosis} | original (display scaled) -> prepared', fill='black')
        draw.text((5, index * 250 + 230), row.id_code, fill='black')
    if len(sample_rows):
        sheet.save(output_dir / 'PRIVATE_before_after.jpg', quality=90)
    print(f'Output: {output_dir}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'artifacts/splits/split_manifest.csv')
    parser.add_argument('--images', type=Path, default=ROOT / 'data/train_images')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/processed_224_v1')
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--crop-black', action='store_true')
    parser.add_argument('--threshold', type=int, default=7)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    run_preprocess(args.manifest, args.images, args.output, args.size,
                   args.crop_black, args.threshold, args.workers, args.limit)


if __name__ == '__main__':
    main()
