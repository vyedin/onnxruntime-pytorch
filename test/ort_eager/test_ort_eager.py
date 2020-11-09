import unittest
import torch

import numpy as np


class TestCustomOps(unittest.TestCase):

    def test_basic(self):
        device = torch.device("ort")
        x = torch.zeros(5, 3, device = device)
        assert(x is None)

if __name__ == '__main__':
    unittest.main()
