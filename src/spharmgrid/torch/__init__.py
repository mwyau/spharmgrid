# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spherical harmonic operations for PyTorch tensors, backed by torch-harmonics.

Import ``spharmgrid.torch`` to load this optional API and its PyTorch dependencies.
"""

from __future__ import annotations

import importlib

try:
    importlib.import_module("torch")
except ModuleNotFoundError as error:
    if error.name == "torch" or (error.name or "").startswith("torch."):
        raise ImportError(
            "spharmgrid.torch requires PyTorch and torch-harmonics; install "
            "both backend dependencies before importing spharmgrid.torch"
        ) from error
    raise

try:
    importlib.import_module("torch_harmonics")
except ModuleNotFoundError as error:
    if error.name == "torch_harmonics" or (error.name or "").startswith(
        "torch_harmonics."
    ):
        raise ImportError(
            "spharmgrid.torch requires torch-harmonics; install a compatible "
            "torch-harmonics build before importing spharmgrid.torch"
        ) from error
    raise

from ._ops import (
    divergence,
    divergent_wind,
    filter,
    gradient,
    helmholtz,
    inverse_gradient,
    inverse_laplacian,
    inverse_vector_laplacian,
    kinematics,
    laplacian,
    potentials,
    regrid,
    regrid_vector,
    rotational_wind,
    streamfunction,
    vector_laplacian,
    velocity_potential,
    vorticity,
    wind,
)

__all__ = [
    "filter",
    "regrid",
    "regrid_vector",
    "gradient",
    "inverse_gradient",
    "laplacian",
    "inverse_laplacian",
    "vector_laplacian",
    "inverse_vector_laplacian",
    "vorticity",
    "divergence",
    "kinematics",
    "streamfunction",
    "velocity_potential",
    "potentials",
    "helmholtz",
    "rotational_wind",
    "divergent_wind",
    "wind",
]
