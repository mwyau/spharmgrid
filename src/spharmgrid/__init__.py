# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Atmospheric spherical harmonic operations for Xarray, PyTorch, and JAX.

Importing :mod:`spharmgrid` registers the ``.sg`` accessors on Xarray
``DataArray`` and ``Dataset`` objects. DUCC through ``ducc0`` performs the
spherical harmonic transforms for the Xarray/NumPy API. The optional
:mod:`spharmgrid.torch` API uses ``torch-harmonics`` for PyTorch tensors, and
:mod:`spharmgrid.jax` uses S2FFT for JAX arrays.
"""

# Import for Xarray accessor registration after direct functions exist.
from . import accessors as _accessors  # noqa: F401
from ._transform import TransformSpec
from .grids import Grid, clenshaw_curtis_grid, detect_grid, gaussian_grid
from .kinematics import (
    divergence,
    divergent_wind,
    helmholtz,
    inverse_vector_laplacian,
    kinematics,
    potentials,
    rotational_wind,
    streamfunction,
    vector_laplacian,
    velocity_potential,
    vorticity,
    wind,
)
from .operators import (
    EARTH_RADIUS_M,
    gradient,
    inverse_gradient,
    inverse_laplacian,
    laplacian,
)
from .regrid import regrid, regrid_vector
from .spectral import filter, parse_spectral

__all__ = [
    "EARTH_RADIUS_M",
    "Grid",
    "TransformSpec",
    "clenshaw_curtis_grid",
    "detect_grid",
    "divergent_wind",
    "divergence",
    "filter",
    "gaussian_grid",
    "gradient",
    "helmholtz",
    "inverse_gradient",
    "inverse_laplacian",
    "inverse_vector_laplacian",
    "kinematics",
    "laplacian",
    "parse_spectral",
    "potentials",
    "regrid",
    "regrid_vector",
    "rotational_wind",
    "streamfunction",
    "velocity_potential",
    "vector_laplacian",
    "vorticity",
    "wind",
]
