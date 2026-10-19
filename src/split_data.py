"""Make reproducible, provisional splits while keeping known duplicates together."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from .common import ROOT, file_hash, save_json


def choose_holdout(frame, fraction, seed):
    """Select the closest class-balanced group split from deterministic candidates."""
    total = np.bincount(frame.diagnosis, minlength=5)
    if (total == 0).any():
        raise ValueError('All five grades must be available before splitting.')
    sampler = GroupShuffleSplit(n_splits=256, test_size=fraction, random_state=seed)
    best = None
    for train, holdout in sampler.split(frame, groups=frame.group_id):
        hold_counts = np.bincount(frame.iloc[holdout].diagnosis, minlength=5)
        train_counts = total - hold_counts
        if (hold_counts == 0).any() or (train_counts == 0).any():
            continue
        score = np.mean(np.abs(hold_counts / total - fraction))
        score += abs(len(holdout) / len(frame) - fraction)
        if best is None or score < best[0]:
            best = score, train, holdout
    if best is None:
        raise ValueError('Unable to preserve every grade across splits. Review group/sample counts.')
    return frame.iloc[best[1]].copy(), frame.iloc[best[2]].copy()


def attach_groups(frame, reviewed_groups=None):
    """Union exact pixel matches and any user-supplied reviewed related-image groups."""
    parent = {i: i for i in frame.id_code}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    for _, rows in frame.groupby('pixel_sha256'):
        ids = list(rows.id_code)
        for image_id in ids[1:]:
            union(ids[0], image_id)
    if reviewed_groups is not None:
        reviewed = pd.read_csv(reviewed_groups, dtype=str)
        if not {'id_code', 'reviewed_group'} <= set(reviewed.columns):
            raise ValueError('Reviewed group CSV must contain id_code and reviewed_group.')
        if reviewed[['id_code', 'reviewed_group']].isna().any().any() or reviewed.id_code.duplicated().any():
            raise ValueError('Reviewed group entries must be non-empty and IDs unique.')
        if not set(reviewed.id_code) <= set(frame.id_code):
            raise ValueError('Reviewed group CSV references unknown/excluded IDs.')
        for _, rows in reviewed.groupby('reviewed_group'):
            ids = list(rows.id_code)
            for image_id in ids[1:]:
                union(ids[0], image_id)
    result = frame.copy()
    result['group_id'] = [find(i) for i in result.id_code]
    return result


def run_split(inventory, destination, seed=2112, reviewed_groups=None):
    frame = pd.read_csv(inventory, dtype={'id_code': str, 'pixel_sha256': str})
    # Conflicting identical-image labels stay out until a documented review resolves them.
    eligible = frame[frame.status.eq('ok') & ~frame.label_conflict].copy()
    eligible = eligible.sort_values('id_code').reset_index(drop=True)
    eligible = attach_groups(eligible, reviewed_groups)
    development, test = choose_holdout(eligible, .15, seed)
    train, validation = choose_holdout(development, .15 / .85, seed + 1)
    pieces = []
    for name, part in [('train', train), ('validation', validation), ('test', test)]:
        part['split'] = name
        pieces.append(part)
    manifest = pd.concat(pieces, ignore_index=True).sort_values('id_code')
    if manifest.groupby('group_id')['split'].nunique().max() != 1:
        raise AssertionError('Related-image group crosses splits.')
    if manifest.groupby('pixel_sha256')['split'].nunique().max() != 1:
        raise AssertionError('Exact pixel duplicate crosses splits.')
    if manifest.id_code.duplicated().any() or len(manifest) != len(eligible):
        raise AssertionError('Coverage or uniqueness check failed.')
    destination.mkdir(parents=True, exist_ok=True)
    columns = ['id_code', 'diagnosis', 'split', 'group_id', 'pixel_sha256']
    manifest[columns].to_csv(destination / 'split_manifest.csv', index=False)
    for name in ['train', 'validation', 'test']:
        manifest.loc[manifest.split.eq(name), columns].to_csv(destination / f'{name}.csv', index=False)
    excluded = frame.loc[~frame.id_code.isin(manifest.id_code), ['id_code', 'diagnosis', 'status', 'label_conflict']]
    excluded.to_csv(destination / 'excluded_records.csv', index=False)
    counts = pd.crosstab(manifest['split'], manifest.diagnosis).reindex(['train', 'validation', 'test'])
    counts.to_csv(destination / 'split_class_counts.csv')
    summary = {'status': 'PROVISIONAL: review similar images and any available patient metadata before final evaluation',
               'seed': seed, 'target_proportions': {'train': .70, 'validation': .15, 'test': .15},
               'actual_counts': manifest.split.value_counts().to_dict(),
               'class_counts': {split: {str(k): int(v) for k, v in row.items()} for split, row in counts.iterrows()},
               'eligible_images': len(manifest), 'excluded_images': len(excluded),
               'groups': int(manifest.group_id.nunique()), 'exact_pixel_overlap_across_splits': 0,
               'inventory_sha256': file_hash(inventory),
               'reviewed_groups_sha256': file_hash(reviewed_groups) if reviewed_groups else None,
               'notes': ['GroupShuffleSplit candidates selected by size and per-class proportional error, not model scores.',
                         'Same exact pixels remain in one split; duplicates are retained, not counted as independent patients.',
                         'No random augmentations or resampling applied to validation/test.',
                         'Freeze reviewed splits before model selection; do not tune on test results.']}
    save_json(destination / 'split_summary.json', summary)
    print(counts.to_string())
    print(f'Wrote {len(manifest)} eligible rows; {len(excluded)} excluded. Splits remain provisional.')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, default=ROOT / 'artifacts/audit/image_inventory.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/splits')
    parser.add_argument('--seed', type=int, default=2112)
    parser.add_argument('--reviewed-groups', type=Path)
    args = parser.parse_args()
    run_split(args.inventory, args.output, args.seed, args.reviewed_groups)


if __name__ == '__main__':
    main()
