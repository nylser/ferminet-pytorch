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

"""Implementation of Fermionic Neural Network in PyTorch."""

import enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import attr
from ferminet_torch import envelopes
from ferminet_torch import network_blocks
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import Protocol


FermiLayers = Tuple[Tuple[int, int], ...]
# Recursive types are not yet supported in pytype - b/109648354.
# pytype: disable=not-supported-yet
ParamTree = Union[
    torch.Tensor, Iterable['ParamTree'], Mapping[Any, 'ParamTree']
]
# pytype: enable=not-supported-yet
# Parameters for a single part of the network are just a dict.
Param = Mapping[str, torch.Tensor]


@attr.s(auto_attribs=True)
class FermiNetData:
    """Data passed to network.

    Shapes given for an unbatched element (i.e. a single MCMC configuration).

    Attributes:
        positions: walker positions, shape (nelectrons*ndim).
        spins: spins of each walker, shape (nelectrons).
        atoms: atomic positions, shape (natoms*ndim).
        charges: atomic charges, shape (natoms).
    """
    positions: Any
    spins: Any
    atoms: Any
    charges: Any


## Interfaces (public) ##


class InitFermiNet(Protocol):
    def __call__(self) -> ParamTree:
        """Returns initialized parameters for the network."""


class FermiNetLike(Protocol):
    def __call__(
        self,
        params: ParamTree,
        electrons: torch.Tensor,
        spins: torch.Tensor,
        atoms: torch.Tensor,
        charges: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the sign and log magnitude of the wavefunction.

        Args:
            params: network parameters.
            electrons: electron positions, shape (nelectrons*ndim), where ndim is the
                dimensionality of the system.
            spins: 1D array specifying the spin state of each electron.
            atoms: positions of nuclei, shape: (natoms, ndim).
            charges: nuclei charges, shape: (natoms).
        """


class LogFermiNetLike(Protocol):
    def __call__(
        self,
        params: ParamTree,
        electrons: torch.Tensor,
        spins: torch.Tensor,
        atoms: torch.Tensor,
        charges: torch.Tensor,
    ) -> torch.Tensor:
        """Returns the log magnitude of the wavefunction.

        Args:
            params: network parameters.
            electrons: electron positions, shape (nelectrons*ndim), where ndim is the
                dimensionality of the system.
            spins: 1D array specifying the spin state of each electron.
            atoms: positions of nuclei, shape: (natoms, ndim).
            charges: nuclear charges, shape: (natoms).
        """


class OrbitalFnLike(Protocol):
    def __call__(
        self,
        params: ParamTree,
        pos: torch.Tensor,
        spins: torch.Tensor,
        atoms: torch.Tensor,
        charges: torch.Tensor,
    ) -> Sequence[torch.Tensor]:
        """Forward evaluation of the Fermionic Neural Network up to the orbitals.

        Args:
            params: network parameter tree.
            pos: The electron positions, a 3N dimensional vector.
            spins: The electron spins, an N dimensional vector.
            atoms: Array with positions of atoms.
            charges: Array with atomic charges.

        Returns:
            Sequence of orbitals.
        """


@attr.s(auto_attribs=True, kw_only=True)
class BaseNetworkOptions:
    """Options controlling the overall network architecture.

    Attributes:
        ndim: dimension of system. Change only with caution.
        determinants: Number of determinants to use.
        states: Number of outputs, one per excited (or ground) state. Ignored if 0.
        full_det: If true, evaluate determinants over all electrons. Otherwise,
            block-diagonalise determinants into spin channels.
        rescale_inputs: If true, rescale the inputs so they grow as log(|r|).
        bias_orbitals: If true, include a bias in the final linear layer to shape
            the outputs into orbitals.
        envelope: Envelope object to create and apply the multiplicative envelope.
        jastrow: Type of Jastrow factor if used, or 'none' if no Jastrow factor.
        complex_output: If true, the network outputs complex numbers.
    """

    ndim: int = 3
    determinants: int = 16
    states: int = 0
    full_det: bool = True
    rescale_inputs: bool = False
    bias_orbitals: bool = False
    envelope: envelopes.Envelope = attr.ib(
        default=attr.Factory(
            envelopes.make_isotropic_envelope,
            takes_self=False))
    jastrow: str = 'none'
    complex_output: bool = False


@attr.s(auto_attribs=True, kw_only=True)
class FermiNetOptions(BaseNetworkOptions):
    """Options controlling the FermiNet architecture.

    Attributes:
        hidden_dims: Tuple of pairs, where each pair contains the number of hidden
            units in the one-electron and two-electron stream in the corresponding
            layer of the FermiNet. The number of layers is given by the length of the
            tuple.
        separate_spin_channels: If True, use separate two-electron streams for
            spin-parallel and spin-antiparallel pairs of electrons. If False, use the
            same stream for all pairs of electrons.
        use_last_layer: If true, the outputs of the one- and two-electron streams
            are combined into permutation-equivariant features and passed into the
            final orbital-shaping layer. Otherwise, just the output of the
            one-electron stream is passed into the orbital-shaping layer.
    """

    hidden_dims: FermiLayers = ((256, 32), (256, 32), (256, 32), (256, 32))
    separate_spin_channels: bool = False
    use_last_layer: bool = False


# Network class.


@attr.s(auto_attribs=True)
class Network:
    options: BaseNetworkOptions
    init: InitFermiNet
    apply: FermiNetLike
    orbitals: OrbitalFnLike


# Internal utilities


def _split_spin_pairs(
    arr: torch.Tensor,
    nspins: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Splits array into parallel and anti-parallel spin channels.

    For an array of dimensions (nelec, nelec, ...), where nelec = sum(nspins),
    and the first nspins[0] elements along the first two axes correspond to the up
    electrons, we have an array like:

        up,up   | up,down
        down,up | down,down

    Split this into the diagonal and off-diagonal blocks. As nspins[0] !=
    nspins[1] in general, flatten the leading two dimensions before combining the
    blocks.

    Args:
        arr: array with leading dimensions (nelec, nelec).
        nspins: number of electrons in each spin channel.

    Returns:
        parallel, antiparallel arrays, where
        - parallel is of shape (nspins[0]**2 + nspins[1]**2, ...) and the first
            nspins[0]**2 elements correspond to the up,up block and the subsequent
            elements to the down,down block.
        - antiparallel is of shape (2 * nspins[0] + nspins[1], ...) and the first
            nspins[0] + nspins[1] elements correspond to the up,down block and the
            subsequent elements to the down,up block.
    """
    if len(nspins) != 2:
        raise ValueError(
            'Separate spin channels has not been verified with spin sampling.'
        )
    up_up, up_down, down_up, down_down = network_blocks.split_into_blocks(
        arr, nspins
    )
    trailing_dims = arr.shape[2:]
    parallel_spins = [
        up_up.reshape((-1,) + trailing_dims),
        down_down.reshape((-1,) + trailing_dims),
    ]
    antiparallel_spins = [
        up_down.reshape((-1,) + trailing_dims),
        down_up.reshape((-1,) + trailing_dims),
    ]
    return (
        torch.cat(parallel_spins, dim=0),
        torch.cat(antiparallel_spins, dim=0),
    )


def _combine_spin_pairs(
    parallel_spins: torch.Tensor,
    antiparallel_spins: torch.Tensor,
    nspins: Tuple[int, int],
) -> torch.Tensor:
    """Combines arrays of parallel spins and antiparallel spins.

    This is the reverse of _split_spin_pairs.

    Args:
        parallel_spins: array of shape (nspins[0]**2 + nspins[1]**2, ...).
        antiparallel_spins: array of shape (2 * nspins[0] * nspins[1], ...).
        nspins: number of electrons in each spin channel.

    Returns:
        array of shape (nelec, nelec, ...).
    """
    if len(nspins) != 2:
        raise ValueError(
            'Separate spin channels has not been verified with spin sampling.'
        )
    nsame_pairs = [nspin**2 for nspin in nspins]
    same_pair_partitions = network_blocks.array_partitions(nsame_pairs)
    up_up, down_down = torch.split(parallel_spins, nsame_pairs, dim=0)
    up_down, down_up = torch.split(antiparallel_spins, 2, dim=0)
    trailing_dims = parallel_spins.shape[1:]
    up = torch.cat(
        (
            up_up.reshape((nspins[0], nspins[0]) + trailing_dims),
            up_down.reshape((nspins[0], nspins[1]) + trailing_dims),
        ),
        dim=1,
    )
    down = torch.cat(
        (
            down_up.reshape((nspins[1], nspins[0]) + trailing_dims),
            down_down.reshape((nspins[1], nspins[1]) + trailing_dims),
        ),
        dim=1,
    )
    return torch.cat((up, down), dim=0)


## Network layers: features ##


def construct_input_features(
    pos: torch.Tensor,
    atoms: torch.Tensor,
    ndim: int = 3) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Constructs inputs to Fermi Net from raw electron and atomic positions.

    Args:
        pos: electron positions. Shape (nelectrons*ndim,).
        atoms: atom positions. Shape (natoms, ndim).
        ndim: dimension of system. Change only with caution.

    Returns:
        ae, ee, r_ae, r_ee tuple, where:
        ae: atom-electron vector. Shape (nelectron, natom, ndim).
        ee: atom-electron vector. Shape (nelectron, nelectron, ndim).
        r_ae: atom-electron distance. Shape (nelectron, natom, 1).
        r_ee: electron-electron distance. Shape (nelectron, nelectron, 1).
        The diagonal terms in r_ee are masked out such that the gradients of these
        terms are also zero.
    """
    assert atoms.shape[1] == ndim
    
    # Reshape pos to get the correct number of electrons
    nelectrons = pos.shape[0] // ndim
    pos = pos.reshape(nelectrons, ndim)
    
    # Calculate atom-electron vectors
    ae = pos.unsqueeze(1) - atoms.unsqueeze(0)  # (nelectron, natom, ndim)
    
    # Calculate electron-electron vectors
    ee = pos.unsqueeze(1) - pos.unsqueeze(0)  # (nelectron, nelectron, ndim)

    # Calculate distances
    r_ae = torch.linalg.norm(ae, dim=2, keepdim=True)
    
    # Avoid computing the norm of zero, as it has undefined grad
    n = ee.shape[0]
    eye = torch.eye(n, device=pos.device)
    
    # Add a small offset to the diagonal to avoid zero distance
    ee_offset = ee + eye.unsqueeze(-1) * 1e-8
    r_ee = torch.linalg.norm(ee_offset, dim=-1) * (1.0 - eye)
    
    return ae, ee, r_ae, r_ee[..., None]


class FermiNetFeatures(nn.Module):
    """Standard features for FermiNet."""
    
    def __init__(self, natoms: int, ndim: int = 3, rescale_inputs: bool = False):
        """Initialize the feature layer.
        
        Args:
            natoms: Number of atoms
            ndim: Dimension of system
            rescale_inputs: Whether to rescale inputs
        """
        super().__init__()
        self.natoms = natoms
        self.ndim = ndim
        self.rescale_inputs = rescale_inputs
        
    def forward(self, ae: torch.Tensor, r_ae: torch.Tensor, 
                ee: torch.Tensor, r_ee: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            ae: Atom-electron vectors
            r_ae: Atom-electron distances
            ee: Electron-electron vectors
            r_ee: Electron-electron distances
            
        Returns:
            One-electron and two-electron features
        """
        if self.rescale_inputs:
            log_r_ae = torch.log(1 + r_ae)  # grows as log(r) rather than r
            ae_features = torch.cat((log_r_ae, ae * log_r_ae / r_ae), dim=2)

            log_r_ee = torch.log(1 + r_ee)
            ee_features = torch.cat((log_r_ee, ee * log_r_ee / r_ee), dim=2)
        else:
            ae_features = torch.cat((r_ae, ae), dim=2)
            ee_features = torch.cat((r_ee, ee), dim=2)
            
        ae_features = ae_features.reshape(ae_features.shape[0], -1)
        return ae_features, ee_features


## Network layers: permutation-equivariance ##


def construct_symmetric_features(
    h_one: torch.Tensor,
    h_two: torch.Tensor,
    nspins: Tuple[int, int],
    h_aux: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Combines intermediate features from rank-one and -two streams.

    Args:
        h_one: set of one-electron features. Shape: (nelectrons, n1), where n1 is
            the output size of the previous layer.
        h_two: set of two-electron features. Shape: (nelectrons, nelectrons, n2),
            where n2 is the output size of the previous layer.
        nspins: Number of spin-up and spin-down electrons.
        h_aux: optional auxiliary features to include. Shape (nelectrons, naux).

    Returns:
        array containing the permutation-equivariant features: the input set of
        one-electron features, the mean of the one-electron features over each
        (occupied) spin channel, and the mean of the two-electron features over each
        (occupied) spin channel. Output shape (nelectrons, 3*n1 + 2*n2 + naux) if
        there are both spin-up and spin-down electrons and
        (nelectrons, 2*n1 + n2 + naux) otherwise.
    """
    # Split features into spin up and spin down electrons
    spin_partitions = network_blocks.array_partitions(nspins)
    h_ones = torch.split(h_one, nspins, dim=0)
    h_twos = torch.split(h_two, nspins, dim=0)

    # Construct inputs to next layer
    # h.size == 0 corresponds to unoccupied spin channels.
    g_one = [torch.mean(h, dim=0, keepdim=True) for h in h_ones if h.numel() > 0]
    g_one = [g.repeat(h_one.shape[0], 1) for g in g_one]

    g_two = [torch.mean(h, dim=0) for h in h_twos if h.numel() > 0]

    features = [h_one] + g_one + g_two
    if h_aux is not None:
        features.append(h_aux)
    return torch.cat(features, dim=1)


## PyTorch implementation of FermiNet ##

class FermiNetLayer(nn.Module):
    """A single layer of the FermiNet."""
    
    def __init__(self, 
                 in_dims_one: int, 
                 out_dims_one: int,
                 in_dims_two: int, 
                 out_dims_two: int,
                 separate_spin_channels: bool = False):
        """Initialize the layer.
        
        Args:
            in_dims_one: Input dimension of one-electron stream
            out_dims_one: Output dimension of one-electron stream
            in_dims_two: Input dimension of two-electron stream
            out_dims_two: Output dimension of two-electron stream
            separate_spin_channels: Whether to use separate spin channels
        """
        super().__init__()
        self.separate_spin_channels = separate_spin_channels
        
        # One-electron stream
        self.linear_one = network_blocks.LinearLayer(in_dims_one, out_dims_one)
        
        # Two-electron stream
        if separate_spin_channels:
            self.linear_two_parallel = network_blocks.LinearLayer(in_dims_two, out_dims_two)
            self.linear_two_antiparallel = network_blocks.LinearLayer(in_dims_two, out_dims_two)
        else:
            self.linear_two = network_blocks.LinearLayer(in_dims_two, out_dims_two)
    
    def forward(self, 
                h_one: torch.Tensor, 
                h_two: Tuple[torch.Tensor, ...],
                nspins: Tuple[int, int]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Forward pass.
        
        Args:
            h_one: One-electron features
            h_two: Two-electron features
            nspins: Number of electrons in each spin channel
            
        Returns:
            Updated one-electron and two-electron features
        """
        # Residual function
        residual = lambda x, y: (x + y) / torch.sqrt(torch.tensor(2.0)) if x.shape == y.shape else y
        
        # One-electron stream
        h_one_next = torch.tanh(self.linear_one(h_one))
        h_one = residual(h_one, h_one_next)
        
        # Two-electron stream
        if self.separate_spin_channels:
            h_two_parallel, h_two_antiparallel = h_two
            h_two_parallel_next = torch.tanh(self.linear_two_parallel(h_two_parallel))
            h_two_antiparallel_next = torch.tanh(self.linear_two_antiparallel(h_two_antiparallel))
            
            h_two_parallel = residual(h_two_parallel, h_two_parallel_next)
            h_two_antiparallel = residual(h_two_antiparallel, h_two_antiparallel_next)
            
            h_two = (h_two_parallel, h_two_antiparallel)
        else:
            h_two_next = torch.tanh(self.linear_two(h_two[0]))
            h_two = (residual(h_two[0], h_two_next),)
        
        return h_one, h_two


class FermiNet(nn.Module):
    """Fermionic Neural Network implementation in PyTorch."""
    
    def __init__(self, 
                 nspins: Tuple[int, int],
                 natoms: int,
                 ndim: int = 3,
                 determinants: int = 16,
                 hidden_dims: FermiLayers = ((256, 32), (256, 32), (256, 32), (256, 32)),
                 separate_spin_channels: bool = False,
                 use_last_layer: bool = False,
                 full_det: bool = True,
                 rescale_inputs: bool = False,
                 bias_orbitals: bool = False,
                 envelope_label: envelopes.EnvelopeLabel = envelopes.EnvelopeLabel.ISOTROPIC,
                 complex_output: bool = False):
        """Initialize the FermiNet.
        
        Args:
            nspins: Number of electrons in each spin channel
            natoms: Number of atoms
            ndim: Dimension of system
            determinants: Number of determinants
            hidden_dims: Hidden dimensions for one-electron and two-electron streams
            separate_spin_channels: Whether to use separate spin channels
            use_last_layer: Whether to use the last layer
            full_det: Whether to use full determinant
            rescale_inputs: Whether to rescale inputs
            bias_orbitals: Whether to use bias in orbitals
            envelope_label: Type of envelope to use
            complex_output: Whether to use complex output
        """
        super().__init__()
        self.nspins = nspins
        self.natoms = natoms
        self.ndim = ndim
        self.determinants = determinants
        self.hidden_dims = hidden_dims
        self.separate_spin_channels = separate_spin_channels
        self.use_last_layer = use_last_layer
        self.full_det = full_det
        self.rescale_inputs = rescale_inputs
        self.bias_orbitals = bias_orbitals
        self.complex_output = complex_output
        
        # Feature layer
        self.feature_layer = FermiNetFeatures(natoms, ndim, rescale_inputs)
        
        # Calculate input dimensions
        num_one_features = natoms * (ndim + 1)
        num_two_features = ndim + 1
        
        # Calculate number of active spin channels
        self.active_spin_channels = [spin for spin in nspins if spin > 0]
        nchannels = len(self.active_spin_channels)
        
        # FermiNet layers
        self.layers = nn.ModuleList()
        dims_one_in = num_one_features
        dims_two_in = num_two_features
        
        for i, (dims_one_out, dims_two_out) in enumerate(hidden_dims):
            # Calculate input dimension for one-electron stream
            if i > 0:
                dims_one_in = nfeatures(dims_one_in, dims_two_in, 0)
            else:
                # First layer has direct input from feature layer
                pass
            
            # Add layer
            self.layers.append(
                FermiNetLayer(
                    in_dims_one=dims_one_in,
                    out_dims_one=dims_one_out,
                    in_dims_two=dims_two_in,
                    out_dims_two=dims_two_out,
                    separate_spin_channels=separate_spin_channels
                )
            )
            
            dims_one_in = dims_one_out
            dims_two_in = dims_two_out
        
        # Calculate output dimensions for orbitals
        if use_last_layer:
            output_dims = nfeatures(dims_one_in, dims_two_in, 0)
        else:
            output_dims = dims_one_in
        
        # Orbital layers
        self.orbital_layers = nn.ModuleList()
        for nspin in self.active_spin_channels:
            if full_det:
                # Dense determinant. Need N orbitals per electron per determinant.
                norbitals = sum(nspins) * determinants
            else:
                # Spin-factored block-diagonal determinant.
                norbitals = nspin * determinants
                
            if complex_output:
                norbitals *= 2  # one output is real, one is imaginary
                
            self.orbital_layers.append(
                network_blocks.LinearLayer(
                    output_dims, norbitals, include_bias=bias_orbitals
                )
            )
        
        # Envelope
        self.envelope = self._create_envelope(envelope_label)
        
    def _create_envelope(self, envelope_label: envelopes.EnvelopeLabel):
        """Create the envelope.
        
        Args:
            envelope_label: Type of envelope to use
            
        Returns:
            Envelope module
        """
        if envelope_label == envelopes.EnvelopeLabel.ISOTROPIC:
            # For simplicity, we'll just implement the isotropic envelope for now
            envelopes_list = nn.ModuleList()
            for nspin in self.active_spin_channels:
                if self.full_det:
                    norbitals = sum(self.nspins) * self.determinants
                else:
                    norbitals = nspin * self.determinants
                    
                if self.complex_output:
                    norbitals = norbitals // 2
                    
                envelopes_list.append(envelopes.IsotropicEnvelope(self.natoms, norbitals))
            return envelopes_list
        else:
            # Implement other envelope types as needed
            raise NotImplementedError(f"Envelope type {envelope_label} not implemented yet")
    
    def forward(self, 
                pos: torch.Tensor, 
                spins: torch.Tensor, 
                atoms: torch.Tensor, 
                charges: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            pos: Electron positions
            spins: Electron spins
            atoms: Atom positions
            charges: Atom charges
            
        Returns:
            Sign and log magnitude of the wavefunction
        """
        # Get orbitals
        orbitals = self.get_orbitals(pos, spins, atoms, charges)
        
        # Calculate determinants
        if self.full_det:
            sign, logdet = network_blocks.logdet_matmul(orbitals)
        else:
            # For spin-factored determinants, we need to calculate determinants separately
            # for each spin channel and multiply them
            signs = []
            logdets = []
            for orbital in orbitals:
                s, ld = network_blocks.logdet_matmul([orbital])
                signs.append(s)
                logdets.append(ld)
            
            sign = torch.prod(torch.stack(signs))
            logdet = torch.sum(torch.stack(logdets))
        
        return sign, logdet
    
    def get_orbitals(self, 
                    pos: torch.Tensor, 
                    spins: torch.Tensor, 
                    atoms: torch.Tensor, 
                    charges: torch.Tensor) -> List[torch.Tensor]:
        """Get orbitals.
        
        Args:
            pos: Electron positions
            spins: Electron spins
            atoms: Atom positions
            charges: Atom charges
            
        Returns:
            List of orbitals
        """
        # Ensure pos is the right shape
        pos = pos.reshape(-1)
        
        # Construct input features
        ae, ee, r_ae, r_ee = construct_input_features(pos, atoms, ndim=self.ndim)
        
        # Get features from feature layer
        h_one, ee_features = self.feature_layer(ae, r_ae, ee, r_ee)
        
        # Process through FermiNet layers
        if self.separate_spin_channels:
            # Split into parallel and antiparallel spin channels
            h_two = _split_spin_pairs(ee_features, self.nspins)
        else:
            # Use the same stream for all electron pairs
            h_two = (ee_features,)
        
        for layer in self.layers:
            # Construct symmetric features for one-electron stream input
            if self.separate_spin_channels:
                h_two_embedding = _combine_spin_pairs(h_two[0], h_two[1], self.nspins)
            else:
                h_two_embedding = h_two[0]
                
            h_one_in = construct_symmetric_features(h_one, h_two_embedding, self.nspins)
            
            # Apply layer
            h_one, h_two = layer(h_one_in, h_two, self.nspins)
        
        # Prepare input to orbital shaping
        if self.use_last_layer:
            if self.separate_spin_channels:
                h_two_embedding = _combine_spin_pairs(h_two[0], h_two[1], self.nspins)
            else:
                h_two_embedding = h_two[0]
                
            h_to_orbitals = construct_symmetric_features(h_one, h_two_embedding, self.nspins)
        else:
            h_to_orbitals = h_one
        
        # Split by spin channels
        h_to_orbitals_channels = torch.split(h_to_orbitals, self.nspins, dim=0)
        h_to_orbitals_channels = [h for h, spin in zip(h_to_orbitals_channels, self.nspins) if spin > 0]
        
        # Create orbitals
        orbitals = []
        for i, h in enumerate(h_to_orbitals_channels):
            orbital = self.orbital_layers[i](h)
            
            # Apply envelope
            if self.envelope is not None:
                ae_channels = torch.split(ae, self.nspins, dim=0)
                r_ae_channels = torch.split(r_ae, self.nspins, dim=0)
                r_ee_channels = torch.split(r_ee, self.nspins, dim=0)
                
                # Only use active spin channels
                active_indices = [j for j, spin in enumerate(self.nspins) if spin > 0]
                ae_channel = ae_channels[active_indices[i]]
                r_ae_channel = r_ae_channels[active_indices[i]]
                r_ee_channel = r_ee_channels[active_indices[i]]
                
                envelope_factor = self.envelope[i](ae_channel, r_ae_channel, r_ee_channel)
                
                # Ensure envelope_factor has the right shape for multiplication
                if envelope_factor.dim() == 1:
                    envelope_factor = envelope_factor.unsqueeze(1)
                
                orbital = orbital * envelope_factor
            
            # Handle complex output
            if self.complex_output:
                orbital = orbital[..., ::2] + 1.0j * orbital[..., 1::2]
            
            # Reshape into matrices
            spin = self.active_spin_channels[i]
            shape = (spin, -1, sum(self.nspins) if self.full_det else spin)
            orbital = orbital.reshape(shape)
            orbital = orbital.transpose(1, 0)
            
            orbitals.append(orbital)
        
        if self.full_det:
            orbitals = [torch.cat(orbitals, dim=1)]
        
        return orbitals


def nfeatures(out1: int, out2: int, aux: int) -> int:
    """Calculate number of features for the one-electron stream.
    
    Args:
        out1: Output dimension of one-electron stream
        out2: Output dimension of two-electron stream
        aux: Output dimension of auxiliary stream
        
    Returns:
        Number of features
    """
    # For simplicity, we assume 2 spin channels (up and down)
    return 3 * out1 + 2 * out2 + aux


def make_fermi_net(
    nspins: Tuple[int, int],
    charges: torch.Tensor,
    *,
    ndim: int = 3,
    determinants: int = 16,
    hidden_dims: FermiLayers = ((256, 32), (256, 32), (256, 32)),
    separate_spin_channels: bool = False,
    use_last_layer: bool = False,
    full_det: bool = True,
    rescale_inputs: bool = False,
    bias_orbitals: bool = False,
    envelope_label: envelopes.EnvelopeLabel = envelopes.EnvelopeLabel.ISOTROPIC,
    complex_output: bool = False,
) -> nn.Module:
    """Creates a FermiNet model.
    
    Args:
        nspins: Number of electrons in each spin channel
        charges: Atom charges
        ndim: Dimension of system
        determinants: Number of determinants
        hidden_dims: Hidden dimensions for one-electron and two-electron streams
        separate_spin_channels: Whether to use separate spin channels
        use_last_layer: Whether to use the last layer
        full_det: Whether to use full determinant
        rescale_inputs: Whether to rescale inputs
        bias_orbitals: Whether to use bias in orbitals
        envelope_label: Type of envelope to use
        complex_output: Whether to use complex output
        
    Returns:
        FermiNet model
    """
    return FermiNet(
        nspins=nspins,
        natoms=charges.shape[0],
        ndim=ndim,
        determinants=determinants,
        hidden_dims=hidden_dims,
        separate_spin_channels=separate_spin_channels,
        use_last_layer=use_last_layer,
        full_det=full_det,
        rescale_inputs=rescale_inputs,
        bias_orbitals=bias_orbitals,
        envelope_label=envelope_label,
        complex_output=complex_output
    )
