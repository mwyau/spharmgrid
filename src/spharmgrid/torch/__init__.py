"""Spherical harmonic operations for PyTorch tensors, backed by torch-harmonics.

Import ``spharmgrid.torch`` to load this optional API and its PyTorch dependencies.
"""

from __future__ import annotations

try:
    import torch as _torch  # noqa: F401
except ModuleNotFoundError as error:
    if error.name == "torch" or (error.name or "").startswith("torch."):
        raise ImportError(
            "spharmgrid.torch requires PyTorch and torch-harmonics; install "
            "the optional dependency with `pip install spharmgrid[torch]`"
        ) from error
    raise

try:
    import torch_harmonics as _torch_harmonics  # noqa: F401
except ModuleNotFoundError as error:
    if error.name == "torch_harmonics" or (error.name or "").startswith(
        "torch_harmonics."
    ):
        raise ImportError(
            "spharmgrid.torch requires torch-harmonics; install the optional "
            "dependency with `pip install spharmgrid[torch]`"
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
