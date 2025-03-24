# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for FermiNet PyTorch implementation."""

import unittest

import torch
from ferminet_torch import networks
from ferminet_torch import envelopes
from ferminet_torch import network_blocks


class NetworkBlocksTest(unittest.TestCase):
    """Tests for network_blocks module."""
    
    def test_linear_layer(self):
        """Test linear layer."""
        x = torch.randn(10, 5)
        w = torch.randn(5, 3)
        b = torch.randn(3)
        
        y = network_blocks.linear_layer(x, w, b)
        
        self.assertEqual(y.shape, (10, 3))
        self.assertTrue(torch.allclose(y, torch.matmul(x, w) + b))
    
    def test_slogdet(self):
        """Test slogdet function."""
        # Test with 1x1 matrix
        x = torch.tensor([[2.0]])
        sign, logdet = network_blocks.slogdet(x)
        self.assertEqual(sign.item(), 1.0)
        self.assertAlmostEqual(logdet.item(), 0.693, places=3)  # log(2) ≈ 0.693
        
        # Test with 2x2 matrix
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        sign, logdet = network_blocks.slogdet(x)
        self.assertEqual(sign.item(), -1.0)
        self.assertAlmostEqual(logdet.item(), 0.693, places=3)  # log(2) ≈ 0.693


class FermiNetTest(unittest.TestCase):
    """Tests for FermiNet implementation."""
    
    def test_ferminet_forward(self):
        """Test FermiNet forward pass."""
        # Define system parameters
        nspins = (1, 1)  # 1 spin-up and 1 spin-down electron
        charges = torch.tensor([1.0])  # Nuclear charge
        
        # Create random electron positions, spins, and atom positions
        pos = torch.randn(sum(nspins) * 3)  # 3D positions for each electron
        spins = torch.tensor([0, 1])  # Spin-up and spin-down
        atoms = torch.zeros(1, 3)  # Atom at origin
        
        # Create FermiNet model
        model = networks.make_fermi_net(
            nspins=nspins,
            charges=charges,
            determinants=4,
            hidden_dims=((32, 16), (32, 16)),
            envelope_label=envelopes.EnvelopeLabel.ISOTROPIC
        )
        
        # Forward pass
        sign, logdet = model(pos, spins, atoms, charges)
        
        # Check output shapes
        self.assertEqual(sign.shape, torch.Size([]))
        self.assertEqual(logdet.shape, torch.Size([]))
        
        # Check that gradients can be computed
        logdet.backward()
        
        # Check that all parameters have gradients
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"Parameter {name} has no gradient")


if __name__ == '__main__':
    unittest.main()
