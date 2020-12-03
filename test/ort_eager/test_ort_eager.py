import unittest
import torch
import torch_ort
import numpy as np


class TestCustomOps(unittest.TestCase):

    def test_basic(self):
        device = torch.device("ort")
        x = torch.empty(5, 3, device = device)
        y = torch.empty(5, 3, device = device)
        z = x + y
        z_pt = x.cpu() + y.cpu()
        assert torch.allclose(z.cpu(), z_pt)
    
if __name__ == '__main__':
    unittest.main()
