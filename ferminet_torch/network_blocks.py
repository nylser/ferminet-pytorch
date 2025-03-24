# Copyright 2022 DeepMind Technologies Limited.
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

"""Neural network building blocks for PyTorch implementation."""

import functools
import itertools
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def array_partitions(sizes: Sequence[int]) -> Sequence[int]:
    """Returns the indices for splitting an array into separate partitions.

    Args:
        sizes: size of each of N partitions. The dimension of the array along
        the relevant axis is assumed to be sum(sizes).

    Returns:
        sequence of indices (length len(sizes)-1) at which an array should be split
        to give the desired partitions.
    """
    return list(itertools.accumulate(sizes))[:-1]


def split_into_blocks(block_arr: torch.Tensor,
                      block_dims: Tuple[int, ...]) -> Sequence[torch.Tensor]:
    """Split a square array into blocks along the leading two axes.

    Consider the (N,N) array
    A B
    C D
    where A=(N1,N1), B=(N1,N2), C=(N2,N1), D=(N2,N2), and N=N1+N2. Split the array
    into the given blocks.

    Args:
        block_arr: block array to split.
        block_dims: the size of each block along each axis (i.e. N1, N2, ...).

    Returns:
        blocks of the array split along the two leading axes into chunks with
        dimensions given by block_dims.
    """
    partitions = array_partitions(block_dims)
    block1 = torch.split(block_arr, block_dims, dim=0)
    block12 = [torch.split(arr, block_dims, dim=1) for arr in block1]
    return tuple(itertools.chain.from_iterable(block12))


def init_linear_layer(
    in_dim: int, out_dim: int, include_bias: bool = True
) -> Dict[str, torch.Tensor]:
    """Initialises parameters for a linear layer, x w + b.

    Args:
        in_dim: input dimension to linear layer.
        out_dim: output dimension (number of hidden units) of linear layer.
        include_bias: if true, include a bias in the linear layer.

    Returns:
        A mapping containing the weight matrix (key 'w') and, if required, bias
        unit (key 'b').
    """
    weight = torch.randn(in_dim, out_dim) / torch.sqrt(torch.tensor(float(in_dim)))
    if include_bias:
        bias = torch.randn(out_dim)
        return {'w': weight, 'b': bias}
    else:
        return {'w': weight}


def linear_layer(x: torch.Tensor,
                 w: torch.Tensor,
                 b: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Evaluates a linear layer, x w + b.

    Args:
        x: inputs.
        w: weights.
        b: optional bias.

    Returns:
        x w + b if b is given, x w otherwise.
    """
    y = torch.matmul(x, w)
    return y + b if b is not None else y


def vmap_linear_layer(x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Vectorized version of linear_layer that applies to batched inputs.

    Args:
        x: batched inputs of shape (batch_size, in_dim)
        w: weights of shape (in_dim, out_dim)
        b: optional bias of shape (out_dim)

    Returns:
        batched outputs of shape (batch_size, out_dim)
    """
    y = torch.matmul(x, w)
    return y + b if b is not None else y


def slogdet(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes sign and log of determinants of matrices.

    This is a torch.linalg.slogdet with a special (fast) path for small matrices.

    Args:
        x: square matrix.

    Returns:
        sign, (natural) logarithm of the determinant of x.
    """
    if x.shape[-1] == 1:
        if torch.is_complex(x):
            sign = x[..., 0, 0] / torch.abs(x[..., 0, 0])
        else:
            sign = torch.sign(x[..., 0, 0])
        logdet = torch.log(torch.abs(x[..., 0, 0]))
    else:
        sign, logdet = torch.linalg.slogdet(x)

    return sign, logdet


def logdet_matmul(
    xs: Sequence[torch.Tensor], w: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Combines determinants and takes dot product with weights in log-domain.

    We use the log-sum-exp trick to reduce numerical instabilities.

    Args:
        xs: FermiNet orbitals in each determinant. Either of length 1 with shape
          (ndet, nelectron, nelectron) (full_det=True) or length 2 with shapes
          (ndet, nalpha, nalpha) and (ndet, nbeta, nbeta) (full_det=False,
          determinants are factorised into block-diagonals for each spin channel).
        w: weight of each determinant. If none, a uniform weight is assumed.

    Returns:
        sum_i w_i D_i in the log domain, where w_i is the weight of D_i, the i-th
        determinant (or product of the i-th determinant in each spin channel, if
        full_det is not used).
    """
    # 1x1 determinants appear to be numerically sensitive and can become 0
    # (especially when multiple determinants are used with the spin-factored
    # wavefunction). Avoid this by not going into the log domain for 1x1 matrices.
    # Pass initial value to functools so det1d = 1 if all matrices are larger than
    # 1x1.
    det1d = functools.reduce(lambda a, b: a * b,
                           [x.reshape(-1) for x in xs if x.shape[-1] == 1], 
                           torch.tensor(1.0, device=xs[0].device))
    
    # Pass initial value to functools so sign_in = 1, logdet = 0 if all matrices
    # are 1x1.
    phase_in, logdet = functools.reduce(
        lambda a, b: (a[0] * b[0], a[1] + b[1]),
        [slogdet(x) for x in xs if x.shape[-1] > 1], 
        (torch.tensor(1.0, device=xs[0].device), torch.tensor(0.0, device=xs[0].device)))

    # log-sum-exp trick
    maxlogdet = torch.max(logdet)
    det = phase_in * det1d * torch.exp(logdet - maxlogdet)
    if w is None:
        result = torch.sum(det)
    else:
        result = torch.matmul(det, w)[0]
    
    # return phase as a unit-norm complex number, rather than as an angle
    if torch.is_complex(result):
        phase_out = torch.angle(result)
    else:
        phase_out = torch.sign(result)
    
    log_out = torch.log(torch.abs(result)) + maxlogdet
    return phase_out, log_out


class LinearLayer(nn.Module):
    """Linear layer with custom initialization."""
    
    def __init__(self, in_dim: int, out_dim: int, include_bias: bool = True):
        """Initialize the linear layer.
        
        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            include_bias: Whether to include bias
        """
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_dim, out_dim) / torch.sqrt(torch.tensor(float(in_dim))))
        self.bias = nn.Parameter(torch.randn(out_dim)) if include_bias else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        y = torch.matmul(x, self.weight)
        return y + self.bias if self.bias is not None else y
