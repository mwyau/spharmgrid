"""Type aliases shared by the Xarray and PyTorch APIs."""

from __future__ import annotations

from typing import Literal, TypeAlias

WindSource: TypeAlias = Literal["vorticity_divergence", "potentials"]
RotationalQuantity: TypeAlias = Literal["vorticity", "streamfunction"]
DivergentQuantity: TypeAlias = Literal["divergence", "velocity_potential"]

__all__ = ["DivergentQuantity", "RotationalQuantity", "WindSource"]
