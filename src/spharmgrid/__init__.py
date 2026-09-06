"""xarray-first spherical-harmonic atmospheric operations using DUCC0.

Importing :mod:`spharmgrid` registers the ``.sg`` accessors on xarray
``DataArray`` and ``Dataset`` objects.  DUCC0 supplies all numerical
spherical-harmonic transforms; spharmgrid supplies the GL/CC, xarray, and CF
operations layer around them.
"""

# Import solely for xarray accessor registration after direct functions exist.
from . import accessors as _accessors  # noqa: F401
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
from .spectral import SpectralRange, filter, parse_spectral

__all__ = [
    "EARTH_RADIUS_M",
    "Grid",
    "SpectralRange",
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
