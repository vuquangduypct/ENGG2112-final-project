import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from src.audit_data import inspect_image, find_near_duplicates
from src.common import read_labels
from src.preprocess import prepare_image, run_preprocess
from src.split_data import attach_groups, run_split


class DataPipelineTests(unittest.TestCase):
    def test_invalid_labels_and_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'train.csv'
            for body in ['id_code,diagnosis\na,5\n', 'id_code,diagnosis\na,1.5\n',
                         'id_code,diagnosis\na,0\na,1\n', 'id_code,diagnosis\n../a,0\n']:
                path.write_text(body)
                with self.assertRaises(ValueError):
                    read_labels(path)

    def test_decoded_duplicates_detected_despite_different_encoding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = Image.fromarray(np.random.default_rng(2).integers(0, 256, (30, 40, 3), dtype=np.uint8))
            image.save(root / 'a.png', compress_level=0)
            image.save(root / 'b.png', compress_level=9)
            a = inspect_image(('a', 0, root / 'a.png'))
            b = inspect_image(('b', 0, root / 'b.png'))
            self.assertNotEqual(a['file_sha256'], b['file_sha256'])
            self.assertEqual(a['pixel_sha256'], b['pixel_sha256'])
            (root / 'bad.png').write_bytes(b'not an image')
            self.assertEqual(inspect_image(('bad', 1, root / 'bad.png'))['status'], 'unreadable')
            self.assertEqual(inspect_image(('missing', 1, root / 'none.png'))['status'], 'missing')

    def test_reviewed_groups_union_transitively_with_exact_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            reviewed = Path(directory) / 'groups.csv'
            pd.DataFrame({'id_code': ['b', 'c'], 'reviewed_group': ['patient1', 'patient1']}).to_csv(reviewed, index=False)
            frame = pd.DataFrame({'id_code': ['a', 'b', 'c', 'd'], 'pixel_sha256': ['same', 'same', 'other', 'last']})
            result = attach_groups(frame, reviewed)
            self.assertEqual(result.group_id.iloc[:3].nunique(), 1)
            self.assertNotEqual(result.group_id.iloc[0], result.group_id.iloc[3])

    def test_split_has_no_overlap_is_reproducible_and_quarantines_conflicts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for grade in range(5):
                for i in range(30):
                    for copy in range(2):
                        rows.append({'id_code': f'g{grade}_{i}_{copy}', 'diagnosis': grade,
                                     'pixel_sha256': f'pixels{grade}_{i}', 'status': 'ok', 'label_conflict': False})
            rows.extend([{'id_code': f'conflict{i}', 'diagnosis': i,
                          'pixel_sha256': 'conflicting', 'status': 'ok', 'label_conflict': True} for i in range(2)])
            inventory = root / 'inventory.csv'
            pd.DataFrame(rows).to_csv(inventory, index=False)
            a = run_split(inventory, root / 'a')
            b = run_split(inventory, root / 'b')
            self.assertEqual(a.groupby('pixel_sha256')['split'].nunique().max(), 1)
            self.assertEqual(len(a), 300)
            self.assertTrue(pd.crosstab(a['split'], a.diagnosis).gt(0).all().all())
            pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))

    def test_resize_preserves_aspect_ratio_and_black_image_survives_crop(self):
        image = Image.new('RGB', (100, 50), (255, 255, 255))
        result, _ = prepare_image(image, 40)
        pixels = np.asarray(result)
        self.assertEqual(result.size, (40, 40))
        self.assertTrue((pixels[0] == 0).all())
        self.assertTrue((pixels[20] == 255).all())
        result, bounds = prepare_image(Image.new('RGB', (30, 20)), 40, True)
        self.assertEqual(bounds, (0, 0, 30, 20))
        self.assertEqual(result.size, (40, 40))

    def test_preprocessing_refuses_to_write_in_originals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                run_preprocess(root / 'missing.csv', root, root / 'processed')


if __name__ == '__main__':
    unittest.main()
