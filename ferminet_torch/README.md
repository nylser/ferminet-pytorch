# FermiNet PyTorch

This is a PyTorch implementation of the Fermionic Neural Network (FermiNet) originally developed by DeepMind in JAX.

## Overview

FermiNet is a neural network architecture designed to represent the ground state wavefunctions of fermionic systems. This PyTorch implementation provides a more accessible version for researchers who are more familiar with PyTorch than JAX.

## Features

- Core FermiNet architecture implemented in PyTorch
- Support for multiple determinants
- Various envelope functions
- Support for complex outputs
- Spin-factored determinants

## Installation

```bash
pip install -e .
```

## Usage

Here's a simple example of how to use the PyTorch implementation:

```python
import torch
from ferminet_torch import networks
from ferminet_torch import envelopes

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
```

## Differences from JAX Implementation

- Uses PyTorch's autograd instead of JAX's transformations
- Implemented as PyTorch modules instead of functional programming style
- Some optimizations specific to JAX have been adapted for PyTorch

## License

Apache License 2.0
