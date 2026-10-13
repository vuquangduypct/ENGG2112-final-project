import tempfile
import unittest
from pathlib import Path
import pandas as pd
from PIL import Image
from src.common import file_hash, save_json
from src.handoff_data import load_manifest, load_split, read_example


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.images = self.root / 'data/processed_224_v1'
        self.images.mkdir(parents=True)
        self.splits = self.root / 'artifacts/splits'
        self.splits.mkdir(parents=True)
        self.rows = [{'id_code': i, 'diagnosis': g, 'split': s, 'group_id': i,
                      'pixel_sha256': i} for i, g, s in [('a', 0, 'train'), ('b', 4, 'validation'), ('c', 2, 'test')]]
        self.prepared = [{**r, 'relative_path': r['id_code'] + '.png'} for r in reversed(self.rows)]
        for i in ['a', 'b', 'c']:
            Image.new('RGB', (224, 224), (20, 40, 60)).save(self.images / f'{i}.png')
        self.write()

    def tearDown(self):
        self.temp.cleanup()

    def write(self):
        path = self.splits / 'split_manifest.csv'
        pd.DataFrame(self.rows).to_csv(path, index=False)
        pd.DataFrame(self.prepared).to_csv(self.images / 'processed_manifest.csv', index=False)
        save_json(self.images / 'preprocessing_config.json',
                  {'manifest_sha256': file_hash(path), 'size': 224, 'mode': 'RGB', 'rows': 3})

    def test_portable_join_uses_id_not_csv_order(self):
        frame = load_split('train', self.root)
        pixels, label = read_example(frame.iloc[0])
        self.assertEqual(label, 0)
        self.assertEqual(pixels.shape, (224, 224, 3))
        self.assertEqual(str(pixels.dtype), 'uint8')
        self.assertEqual(frame.iloc[0].id_code, 'a')

    def test_mismatched_label_rejected(self):
        self.prepared[0]['diagnosis'] = 4
        self.write()
        with self.assertRaisesRegex(ValueError, 'disagree'):
            load_manifest(self.root)

    def test_unsafe_filename_rejected(self):
        self.prepared[0]['relative_path'] = '../../outside.png'
        self.write()
        with self.assertRaisesRegex(ValueError, 'filename'):
            load_manifest(self.root)

    def test_cross_split_duplicate_rejected(self):
        self.rows[1]['pixel_sha256'] = self.rows[0]['pixel_sha256']
        self.write()
        with self.assertRaisesRegex(ValueError, 'crosses'):
            load_manifest(self.root)

    def test_missing_image_rejected(self):
        (self.images / 'a.png').unlink()
        with self.assertRaises(FileNotFoundError):
            load_manifest(self.root)

    def test_test_split_reserved_by_default(self):
        with self.assertRaisesRegex(ValueError, 'reserved'):
            load_split('test', self.root)


if __name__ == '__main__':
    unittest.main()
