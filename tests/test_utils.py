"""
    Source: https://github.com/vlukiyanov/pt-dec/blob/master/tests/test_utils.py

"""
from utils import target_distribution
from unittest import TestCase
import torch

class TestTargetDistribution(TestCase):
    def test_basic(self):
        """
        Basic test to check that the calculation is sensible and conforms to the formula.
        """
        test_tensor = torch.Tensor([[0.5, 0.5], [0.0, 1.0]])
        output = target_distribution(test_tensor)
        self.assertAlmostEqual(tuple(output[0]), (0.75, 0.25))
        self.assertAlmostEqual(tuple(output[1]), (0.0, 1.0))
