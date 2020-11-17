import unittest
import torch
import torch_ort
import numpy as np


class TestCustomOps(unittest.TestCase):

    def test_basic(self):
        device = torch.device("ort")
        x = torch.empty(5, 3, device = device)
        #TODO: enable the test later
        #assert(x is None)

if __name__ == '__main__':
    unittest.main()
