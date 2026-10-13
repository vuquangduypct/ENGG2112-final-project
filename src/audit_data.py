"""Read-only image audit; all outputs are private local artifacts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import platform
import sys
import warnings
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw, __version__ as pillow_version
from .common import ROOT, CLASS_NAMES, difference_hash, file_hash, read_labels, save_json


def inspect_image(row):
    image_id, label, path = row
    result = {'id_code': image_id, 'diagnosis': label, 'path': str(path),
              'status': 'ok', 'error': ''}
    if not path.is_file():
        return {**result, 'status': 'missing', 'error': 'Image file not found'}
    try:
        result['bytes'] = path.stat().st_size
        result['file_sha256'] = file_hash(path)
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(path) as source:
                source.verify()  # verify PNG integrity, including CRCs
            with Image.open(path) as source:
                source.load()  # force full decoding rather than just reading the header
                result.update(width=source.width, height=source.height, mode=source.mode)
                rgb = source.convert('RGB')
                result['pixel_sha256'] = hashlib.sha256(
                    f'{rgb.width}x{rgb.height}:RGB:'.encode() + rgb.tobytes()).hexdigest()
                result['dhash'] = difference_hash(rgb)
                thumb = np.asarray(rgb.resize((64, 64))).astype(float)
                result['mean_intensity'] = float(thumb.mean())
                result['intensity_std'] = float(thumb.std())
                result['dark_pixel_fraction'] = float((thumb.max(axis=2) <= 10).mean())
    except Exception as error:
        result.update(status='unreadable', error=f'{type(error).__name__}: {error}')
    return result


def find_near_duplicates(frame, max_distance=2):
    """List visually similar candidates for human review. Do not change labels."""
    records = frame.to_dict('records')
    found = []
    for i, left in enumerate(records):
        left_hash = int(left['dhash'], 16)
        for right in records[i + 1:]:
            if left['pixel_sha256'] == right['pixel_sha256']:
                continue
            distance = (left_hash ^ int(right['dhash'], 16)).bit_count()
            if distance <= max_distance:
                found.append({'left_id': left['id_code'], 'right_id': right['id_code'],
                              'left_label': left['diagnosis'], 'right_label': right['diagnosis'],
                              'dhash_distance': distance, 'review_decision': ''})
    columns = ['left_id', 'right_id', 'left_label', 'right_label', 'dhash_distance', 'review_decision']
    return pd.DataFrame(found, columns=columns).sort_values('dhash_distance')


def make_plots(labels, images, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    counts = labels.diagnosis.value_counts().reindex(range(5), fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar([f'{k}: {CLASS_NAMES[k]}' for k in range(5)], counts, color='#237a91')
    ax.bar_label(bars, padding=3)
    ax.set(ylabel='Number of labelled images', title='APTOS 2019: original five-grade labels')
    ax.tick_params(axis='x', labelsize=8)
    ax.set_ylim(0, max(counts) * 1.14)
    fig.tight_layout()
    fig.savefig(destination / 'class_distribution.png', dpi=170)
    plt.close(fig)
    good = images[images.status.eq('ok')]
    if not good.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(good.width, good.height, s=8, alpha=.3)
        ax.set(xlabel='Width (pixels)', ylabel='Height (pixels)', title='Original image dimensions')
        fig.tight_layout()
        fig.savefig(destination / 'image_dimensions.png', dpi=170)
        plt.close(fig)
        sheet = Image.new('RGB', (3 * 220, 5 * 250), 'white')
        draw = ImageDraw.Draw(sheet)
        for label in range(5):
            for col, row in enumerate(good[good.diagnosis.eq(label)].head(3).itertuples()):
                with Image.open(row.path) as source:
                    sample = ImageOps.pad(source.convert('RGB'), (210, 210))
                x, y = col * 220, label * 250
                sheet.paste(sample, (x, y))
                draw.text((x + 3, y + 214), f'{label}: {CLASS_NAMES[label]}', fill='black')
                draw.text((x + 3, y + 229), row.id_code, fill='black')
        sheet.save(destination / 'PRIVATE_sample_images.jpg', quality=90)


def run_audit(labels_path, image_dir, output_dir, workers=4, near_distance=2):
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = read_labels(labels_path)
    jobs = [(r.id_code, int(r.diagnosis), image_dir / f'{r.id_code}.png')
            for r in labels.itertuples()]
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for number, result in enumerate(pool.map(inspect_image, jobs), 1):
            results.append(result)
            if number % 250 == 0 or number == len(jobs):
                print(f'Checked {number}/{len(jobs)} images', flush=True)
    images = pd.DataFrame(results)
    for column in ['pixel_sha256', 'dhash']:
        if column not in images:
            images[column] = ''
    images['duplicate_group'] = images['pixel_sha256']
    good = images[images.status.eq('ok')].copy()
    exact = good[good.pixel_sha256.duplicated(keep=False)].copy()
    conflicts = set(good.groupby('pixel_sha256').diagnosis.nunique().loc[lambda s: s > 1].index)
    images['label_conflict'] = images.pixel_sha256.isin(conflicts)
    images['eligible'] = images.status.eq('ok') & ~images.label_conflict
    images.to_csv(output_dir / 'image_inventory.csv', index=False)
    exact.to_csv(output_dir / 'exact_duplicates.csv', index=False)
    images[images.label_conflict].to_csv(output_dir / 'conflicting_labels.csv', index=False)
    near = find_near_duplicates(good, near_distance)
    near.to_csv(output_dir / 'near_duplicate_candidates.csv', index=False)
    expected = {f'{i}.png' for i in labels.id_code}
    present = {p.name for p in image_dir.glob('*.png')}
    counts = labels.diagnosis.value_counts().reindex(range(5), fill_value=0)
    class_table = pd.DataFrame({'grade': range(5), 'name': list(CLASS_NAMES.values()),
                               'count': counts.values, 'percent': (counts.values / len(labels) * 100)})
    class_table.to_csv(output_dir / 'class_distribution.csv', index=False)
    summary = {'label_rows': len(labels), 'label_columns': list(labels.columns),
               'labels_sha256': file_hash(labels_path), 'image_directory': str(image_dir.resolve()),
               'readable_images': len(good), 'missing_images': int(images.status.eq('missing').sum()),
               'unreadable_images': int(images.status.eq('unreadable').sum()),
               'extra_png_files': sorted(present - expected),
               'class_counts': {str(k): int(v) for k, v in counts.items()},
               'exact_duplicate_groups': int(exact.pixel_sha256.nunique()),
               'images_in_exact_duplicate_groups': len(exact),
               'conflicting_label_groups': len(conflicts),
               'images_with_conflicting_labels': int(images.label_conflict.sum()),
               'eligible_before_manual_review': int(images.eligible.sum()),
               'near_duplicate_candidate_pairs': len(near), 'dhash_distance_threshold': near_distance,
               'age_metadata_present_in_label_csv': any('age' == c.lower() for c in labels.columns),
               'patient_id_present_in_label_csv': any(c.lower() in {'patient', 'patient_id'} for c in labels.columns),
               'limitations': ['No claim of patient-level separation without patient metadata.',
                              'dHash candidates require manual review; candidate absence does not prove independence.',
                              'Conflicting exact-image labels are quarantined from splits, not corrected.',
                              'No expert assessment of clinical image quality or label correctness was performed.'],
               'environment': {'python': sys.version, 'platform': platform.platform(),
                               'pandas': pd.__version__, 'numpy': np.__version__, 'Pillow': pillow_version}}
    save_json(output_dir / 'audit_summary.json', summary)
    make_plots(labels, images, output_dir)
    print(summary, flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, default=ROOT / 'data/train.csv')
    parser.add_argument('--images', type=Path, default=ROOT / 'data/train_images')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/audit')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--near-distance', type=int, default=2)
    args = parser.parse_args()
    if not 0 <= args.near_distance <= 64 or args.workers < 1:
        parser.error('Invalid workers or dHash distance.')
    run_audit(args.labels, args.images, args.output, args.workers, args.near_distance)


if __name__ == '__main__':
    main()
