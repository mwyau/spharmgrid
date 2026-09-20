# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spherical harmonic operations for JAX arrays, backed by S2FFT.

Import ``spharmgrid.jax`` to load this optional API and its JAX and S2FFT
dependencies.
"""

from __future__ import annotations

import importlib

try:
    importlib.import_module("jax")
except ModuleNotFoundError as error:
    if error.name == "jax" or (error.name or "").startswith("jax."):
        raise ImportError(
            "spharmgrid.jax requires JAX and S2FFT; install the 'jax' extra "
            "before importing spharmgrid.jax"
        ) from error
    raise

try:
    importlib.import_module("s2fft")
except ModuleNotFoundError as error:
    if error.name == "s2fft" or (error.name or "").startswith("s2fft."):
        raise ImportError(
            "spharmgrid.jax requires S2FFT; install the 'jax' extra before "
            "importing spharmgrid.jax"
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
