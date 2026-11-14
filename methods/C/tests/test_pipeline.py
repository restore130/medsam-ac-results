import unittest
from pathlib import Path

import numpy as np

from baseline_inference import segmentation_metrics
from medsam_dataset import KvasirMedSAMDataset, create_dataloader


PROJECT_DIR = Path(__file__).resolve().parents[1]


class MetricsTests(unittest.TestCase):
    def test_perfect_prediction(self):
        target = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        self.assertEqual(segmentation_metrics(target, target), (1.0, 1.0))

    def test_disjoint_prediction(self):
        pred = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        target = np.array([[0, 1], [0, 0]], dtype=np.uint8)
        self.assertEqual(segmentation_metrics(pred, target), (0.0, 0.0))


class DatasetTests(unittest.TestCase):
    def test_test_split_and_shapes(self):
        dataset = KvasirMedSAMDataset(PROJECT_DIR, "test")
        self.assertEqual(len(dataset), 150)
        sample = dataset[0]
        self.assertEqual(tuple(sample["image"].shape), (3, 1024, 1024))
        self.assertEqual(tuple(sample["bbox_1024"].shape), (4,))
        self.assertEqual(str(sample["image"].dtype), "torch.float32")

    def test_loader_rejects_variable_size_batches(self):
        with self.assertRaises(ValueError):
            create_dataloader(PROJECT_DIR, "test", batch_size=2)


if __name__ == "__main__":
    unittest.main()
