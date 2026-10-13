from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = {0: 'No DR', 1: 'Mild', 2: 'Moderate', 3: 'Severe', 4: 'Proliferative DR'}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def read_labels(path):
    frame = pd.read_csv(path, dtype={'id_code': 'string'})
    missing = {'id_code', 'diagnosis'} - set(frame.columns)
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    ids = frame['id_code']
    if ids.isna().any() or not ids.str.fullmatch(r'[A-Za-z0-9_-]+').all():
        raise ValueError('Missing or unsafe image ID. Review labels before proceeding.')
    if ids.duplicated().any():
        raise ValueError('Duplicate image IDs in CSV. Resolve explicitly; no rows were dropped.')
    numeric = pd.to_numeric(frame['diagnosis'], errors='coerce')
    if not numeric.isin(range(5)).all():
        raise ValueError('Labels must be integers 0, 1, 2, 3, 4; no binary remapping is used.')
    frame['diagnosis'] = numeric.astype(int)
    return frame


def difference_hash(image):
    """64-bit dHash: only a review heuristic, NOT a patient identifier."""
    from PIL import Image
    small = np.asarray(image.convert('L').resize((9, 8), Image.Resampling.LANCZOS))
    bits = (small[:, 1:] > small[:, :-1]).ravel()
    return f'{int.from_bytes(np.packbits(bits).tobytes(), "big"):016x}'
