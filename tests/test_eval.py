from eval import cluster_accuracy, entropy, balance
import numpy as np
from unittest import TestCase
import torch

class TestClusterAccuracy(TestCase):
    def test_basic(self):
        """
        Source: https://github.com/vlukiyanov/pt-dec/blob/master/tests/test_utils.py
        Basic test to check that the calculation is sensible.
        """
        true_value1 = np.array([1, 2, 1, 2, 0, 0], dtype=np.int64)
        pred_value1 = np.array([2, 1, 2, 1, 0, 0], dtype=np.int64)
        self.assertAlmostEqual(cluster_accuracy(true_value1, pred_value1)[1], 1.0)
        self.assertAlmostEqual(cluster_accuracy(true_value1, pred_value1, 3)[1], 1.0)
        self.assertDictEqual(
            cluster_accuracy(true_value1, pred_value1)[0], {0: 0, 1: 2, 2: 1}
        )
        true_value2 = np.array([1, 1, 1, 1, 1, 1], dtype=np.int64)
        pred_value2 = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
        self.assertAlmostEqual(cluster_accuracy(true_value2, pred_value2)[1], 1.0 / 6.0)
        self.assertAlmostEqual(
            cluster_accuracy(true_value2, pred_value2, 6)[1], 1.0 / 6.0
        )
        true_value3 = np.array([1, 3, 1, 3, 0, 2], dtype=np.int64)
        pred_value3 = np.array([2, 1, 2, 1, 3, 0], dtype=np.int64)
        self.assertDictEqual(
            cluster_accuracy(true_value3, pred_value3)[0], {2: 1, 1: 3, 3: 0, 0: 2}
        )

class TestEntropy(TestCase):
    def test_entropy(self):
        """
        Basic test to check that the calculation is sensible.
        """
        test_tensor = torch.tensor([1, 2, 3, 4, 5])
        true_value = -18.274547576904297
        test_value = entropy(test_tensor)
        self.assertAlmostEqual(true_value, test_value.item())

