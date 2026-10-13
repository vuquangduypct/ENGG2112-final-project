"""Portable data interface for teammate B. No ML framework is required."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from .common import ROOT, file_hash


def load_manifest(root=ROOT):
    """Join labels and image paths by ID, never by directory order."""
    root = Path(root)
    split_path = root / 'artifacts/splits/split_manifest.csv'
    image_dir = root / 'data/processed_224_v1'
    manifest = pd.read_csv(split_path, dtype={'id_code': str})
    prepared = pd.read_csv(image_dir / 'processed_manifest.csv', dtype={'id_code': str})
    if not {'id_code', 'diagnosis', 'split', 'group_id', 'pixel_sha256'} <= set(manifest.columns):
        raise ValueError('Split manifest lacks required columns.')
    if not {'id_code', 'diagnosis', 'split', 'relative_path'} <= set(prepared.columns):
        raise ValueError('Prepared manifest lacks required columns.')
    for frame in (manifest, prepared):
        if frame.id_code.isna().any() or not frame.id_code.str.fullmatch(r'[A-Za-z0-9_-]+').all():
            raise ValueError('Missing or unsafe image ID.')
        if frame.id_code.duplicated().any() or not frame.diagnosis.isin(range(5)).all():
            raise ValueError('Repeated ID or invalid diagnosis.')
        if not frame.split.isin(['train', 'validation', 'test']).all():
            raise ValueError('Invalid split name.')
    if manifest[['group_id', 'pixel_sha256']].isna().any().any():
        raise ValueError('Missing duplicate group information.')
    if set(manifest.id_code) != set(prepared.id_code):
        raise ValueError('Prepared and split manifests contain different IDs.')
    for key in ('group_id', 'pixel_sha256'):
        if manifest.groupby(key)['split'].nunique().max() != 1:
            raise ValueError(f'{key} crosses split boundaries.')
    merged = manifest.merge(prepared[['id_code', 'diagnosis', 'split', 'relative_path']],
                            on='id_code', validate='one_to_one', suffixes=('', '_prepared'))
    if not merged.diagnosis.eq(merged.diagnosis_prepared).all() or not merged.split.eq(merged.split_prepared).all():
        raise ValueError('Prepared labels/splits disagree with the split manifest.')
    if not merged.relative_path.eq(merged.id_code + '.png').all():
        raise ValueError('Unexpected image filename; paths must be ID.png.')
    config = json.loads((image_dir / 'preprocessing_config.json').read_text(encoding='utf-8'))
    if config['manifest_sha256'] != file_hash(split_path):
        raise ValueError('Prepared images were made with a different split manifest.')
    if config['rows'] != len(merged) or config['size'] != 224 or config['mode'] != 'RGB':
        raise ValueError('Unexpected preprocessing configuration for this handoff.')
    merged['image_path'] = [str(image_dir / name) for name in merged.relative_path]
    if not all(Path(path).is_file() for path in merged.image_path):
        raise FileNotFoundError('Some prepared images are missing; extract the data pack first.')
    return merged.drop(columns=['diagnosis_prepared', 'split_prepared'])


def load_split(split='train', root=ROOT, allow_test=False):
    if split not in {'train', 'validation', 'test'}:
        raise ValueError('Use train, validation or test.')
    if split == 'test' and not allow_test:
        raise ValueError('Test is reserved. Review/freeze the split before final evaluation; do not tune on test.')
    manifest = load_manifest(root)
    return manifest[manifest.split.eq(split)].reset_index(drop=True)


def read_example(row):
    """Return HWC RGB uint8 pixels and integer grade; normalization is B's responsibility."""
    with Image.open(row.image_path) as image:
        if image.mode != 'RGB' or image.size != (224, 224):
            raise ValueError('Expected prepared 224x224 RGB image.')
        pixels = np.array(image)
    return pixels, int(row.diagnosis)


def verify(root=ROOT):
    root = Path(root)
    contract = json.loads((root / 'docs/handoff_contract.json').read_text(encoding='utf-8'))
    for relative, expected in contract['metadata_sha256'].items():
        path = (root / relative).resolve()
        if root.resolve() not in path.parents:
            raise ValueError('Contract path is outside the project.')
        if file_hash(path) != expected:
            raise ValueError(f'Changed/mismatched metadata: {relative}')
    frame = load_manifest(root)
    if frame.split.value_counts().to_dict() != contract['split_counts']:
        raise ValueError('Split counts do not match the handoff version.')
    checks = pd.read_csv(root / 'artifacts/handoff/image_checksums.csv', dtype=str)
    if checks.id_code.duplicated().any() or set(checks.id_code) != set(frame.id_code):
        raise ValueError('Image checksum inventory has different IDs.')
    hashes = checks.set_index('id_code').sha256.to_dict()
    def check(row):
        if file_hash(row.image_path) != hashes[row.id_code]:
            raise ValueError(f'Changed prepared image: {row.id_code}')
        pixels, grade = read_example(row)
        if pixels.dtype != np.uint8 or grade not in range(5):
            raise ValueError('Unexpected pixel type or grade.')
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(check, frame.itertuples()))
    result = {'version': contract['version'], 'status': contract['status'],
              'verified_images': len(frame), 'split_counts': contract['split_counts'],
              'exact_duplicate_overlap': 0, 'manual_similarity_review_complete': False}
    print(json.dumps(result, indent=2))
    print('PASS: file hashes, image decoding, IDs, grades, split assignments and duplicate groups.')
    print('This verifies the handoff files, not clinical labels or patient independence.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        verify(args.root)
    else:
        frame = load_split('train', args.root)
        pixels, label = read_example(frame.iloc[0])
        print(f'Train records: {len(frame)}; first image: shape={pixels.shape}, dtype={pixels.dtype}, grade={label}')
        print('RGB pixels remain 0..255. Add the normalization required by your chosen model once.')


if __name__ == '__main__':
    main()
