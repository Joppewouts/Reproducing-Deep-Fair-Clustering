"""
    Source: https://github.com/vlukiyanov/pt-dec/blob/master/tests/test_cluster.py

"""

import torch
from unittest import TestCase
from module import ClusterAssignment


class TestClusterAssignment(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cluster = ClusterAssignment(
            cluster_number=2,
            embedding_dimension=2,
            cluster_centers=torch.Tensor([[-1, -1], [1, 1]]).float(),
            alpha=1,
        )

    def test_calculation(self):
        test_tensor = torch.Tensor([-2, -2]).float().unsqueeze(0)
        den = float(1) / 3 + float(1) / 19
        gold = torch.Tensor([(float(1) / 3) / den, (float(1) / 19) / den])
        output = self.cluster(test_tensor).data
        self.assertAlmostEqual((gold - output).numpy()[0][0], 0.0)
        self.assertAlmostEqual((gold - output).numpy()[0][1], 0.0)

