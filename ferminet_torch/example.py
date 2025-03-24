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

"""Example usage of FermiNet in PyTorch."""

import torch
from ferminet_torch import networks
from ferminet_torch import envelopes


def main():
    """Run a simple example of FermiNet in PyTorch."""
    # Define system parameters
    nspins = (1, 1)  # 1 spin-up and 1 spin-down electron
    natoms = 1  # 1 atom (hydrogen molecule)
    charges = torch.tensor([1.0])  # Nuclear charge
    
    # Create random electron positions, spins, and atom positions
    pos = torch.randn(sum(nspins) * 3)  # 3D positions for each electron
    spins = torch.tensor([0, 1])  # Spin-up and spin-down
    atoms = torch.zeros(natoms, 3)  # Atom at origin
    
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
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Sign: {sign}")
    print(f"Log determinant: {logdet}")
    
    # Compute gradients
    logdet.backward()
    
    # Print parameter gradients
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}: grad shape {param.grad.shape}, grad norm {param.grad.norm()}")


if __name__ == "__main__":
    main()
