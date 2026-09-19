# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reusable ``torch.nn.Module`` wrappers for spharmgrid Torch operations."""

from __future__ import annotations

import torch
from torch import Tensor

from ..._kinematics_types import (
    DivergentWindSource,
    RotationalWindSource,
    WindSource,
)
from ..._transform import TransformSpec
from ...grids import Grid
from ...operators import EARTH_RADIUS_M
from ...spectral import _resolve_spectral_spec, _validate_taper
from .._backend import (
    _make_state,
    _require_tensor,
    _require_vector_bandwidth,
    _require_vector_tensors,
    _validate_grid,
    _validate_radius,
)
from .._ops import (
    _filter_impl,
    _gradient_with_state,
    _helmholtz_with_state,
    _inverse_gradient_with_state,
    _inverse_laplacian_with_state,
    _kinematics_with_state,
    _laplacian_with_state,
    _potentials_with_state,
    _regrid_impl,
    _regrid_vector_impl,
    _single_source_wind_with_state,
    _validate_scalar_source,
    _validate_source,
    _vector_laplacian_with_state,
    _wind_with_state,
)


class SHTOperators(torch.nn.Module):
    """Reusable same-grid spherical harmonic operations.

    The object owns one set of scalar and vector transform modules.  Move it to
    the input tensor's device with ``.to(device)`` when it is used in a model.
    """

    def __init__(
        self,
        grid: Grid,
        *,
        radius: float = EARTH_RADIUS_M,
    ) -> None:
        super().__init__()
        _validate_grid(grid)
        _validate_radius(radius)
        self.grid = grid
        self.radius = radius
        self._state = _make_state(
            grid,
            grid,
            None,
            vector=True,
            device=torch.device("cpu"),
        )

    def filter(
        self,
        field: Tensor,
        truncation: str | TransformSpec | None = None,
        *,
        lmin: int | None = None,
        lmax: int | None = None,
        taper: float | None = None,
    ) -> Tensor:
        selection = _resolve_spectral_spec(truncation, lmin=lmin, lmax=lmax)
        _validate_taper(taper)
        _require_tensor(field, self.grid)
        return _filter_impl(
            field,
            self._state,
            selection or self._state.spec,
            taper,
        )

    def gradient(self, field: Tensor) -> tuple[Tensor, Tensor]:
        _require_tensor(field, self.grid)
        return _gradient_with_state(field, self._state, self.radius)

    def inverse_gradient(self, eastward: Tensor, northward: Tensor) -> Tensor:
        _require_vector_tensors(eastward, northward, self.grid)
        return _inverse_gradient_with_state(
            eastward,
            northward,
            self._state,
            self.radius,
        )

    def laplacian(self, field: Tensor) -> Tensor:
        _require_tensor(field, self.grid)
        return _laplacian_with_state(field, self._state, self.radius)

    def inverse_laplacian(self, field: Tensor) -> Tensor:
        _require_tensor(field, self.grid)
        return _inverse_laplacian_with_state(field, self._state, self.radius)

    def vector_laplacian(self, u: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        _require_vector_tensors(u, v, self.grid)
        return _vector_laplacian_with_state(
            u,
            v,
            self._state,
            self.radius,
            False,
        )

    def inverse_vector_laplacian(
        self,
        u: Tensor,
        v: Tensor,
    ) -> tuple[Tensor, Tensor]:
        _require_vector_tensors(u, v, self.grid)
        return _vector_laplacian_with_state(
            u,
            v,
            self._state,
            self.radius,
            True,
        )

    def vorticity(self, u: Tensor, v: Tensor) -> Tensor:
        _require_vector_tensors(u, v, self.grid)
        return _kinematics_with_state(u, v, self._state, self.radius)[0]

    def divergence(self, u: Tensor, v: Tensor) -> Tensor:
        _require_vector_tensors(u, v, self.grid)
        return _kinematics_with_state(u, v, self._state, self.radius)[1]

    def kinematics(self, u: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        _require_vector_tensors(u, v, self.grid)
        return _kinematics_with_state(u, v, self._state, self.radius)

    def streamfunction(self, u: Tensor, v: Tensor) -> Tensor:
        _require_vector_tensors(u, v, self.grid)
        return _potentials_with_state(u, v, self._state, self.radius)[0]

    def velocity_potential(self, u: Tensor, v: Tensor) -> Tensor:
        _require_vector_tensors(u, v, self.grid)
        return _potentials_with_state(u, v, self._state, self.radius)[1]

    def potentials(self, u: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        _require_vector_tensors(u, v, self.grid)
        return _potentials_with_state(u, v, self._state, self.radius)

    def helmholtz(self, u: Tensor, v: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        _require_vector_tensors(u, v, self.grid)
        return _helmholtz_with_state(u, v, self._state)

    def rotational_wind(
        self,
        field: Tensor,
        *,
        source: RotationalWindSource,
    ) -> tuple[Tensor, Tensor]:
        _validate_scalar_source(source, ("vorticity", "streamfunction"))
        _require_tensor(field, self.grid)
        return _single_source_wind_with_state(
            field,
            self._state,
            source,
            "rotational",
            self.radius,
        )

    def divergent_wind(
        self,
        field: Tensor,
        *,
        source: DivergentWindSource,
    ) -> tuple[Tensor, Tensor]:
        _validate_scalar_source(
            source,
            ("divergence", "velocity_potential"),
        )
        _require_tensor(field, self.grid)
        return _single_source_wind_with_state(
            field,
            self._state,
            source,
            "divergent",
            self.radius,
        )

    def wind(
        self,
        first: Tensor,
        second: Tensor,
        *,
        source: WindSource,
    ) -> tuple[Tensor, Tensor]:
        _validate_source(source)
        _require_vector_tensors(first, second, self.grid)
        return _wind_with_state(first, second, self._state, source, self.radius)


class SHTFilter(torch.nn.Module):
    """Fixed spectral filter layer using one reusable scalar transform pair."""

    def __init__(
        self,
        grid: Grid,
        truncation: str | TransformSpec | None = None,
        *,
        lmin: int | None = None,
        lmax: int | None = None,
        taper: float | None = None,
    ) -> None:
        super().__init__()
        _validate_grid(grid)
        selection = _resolve_spectral_spec(truncation, lmin=lmin, lmax=lmax)
        _validate_taper(taper)
        self.grid = grid
        self.truncation = selection
        self.taper = taper
        self._state = _make_state(
            grid,
            grid,
            selection,
            vector=False,
            device=torch.device("cpu"),
        )

    def forward(self, field: Tensor) -> Tensor:
        _require_tensor(field, self.grid)
        return _filter_impl(
            field,
            self._state,
            self._state.spec,
            self.taper,
        )


class SHTRegrid(torch.nn.Module):
    """Fixed scalar spectral regridding layer."""

    def __init__(
        self,
        source_grid: Grid,
        target_grid: Grid,
        truncation: str | TransformSpec | None = None,
        *,
        lmin: int | None = None,
        lmax: int | None = None,
        taper: float | None = None,
    ) -> None:
        super().__init__()
        _validate_grid(source_grid, "source_grid")
        _validate_grid(target_grid, "target_grid")
        selection = _resolve_spectral_spec(truncation, lmin=lmin, lmax=lmax)
        _validate_taper(taper)
        self.source_grid = source_grid
        self.target_grid = target_grid
        self.truncation = selection
        self.taper = taper
        self._state = _make_state(
            source_grid,
            target_grid,
            selection,
            vector=False,
            device=torch.device("cpu"),
        )

    def forward(self, field: Tensor) -> Tensor:
        _require_tensor(field, self.source_grid)
        return _regrid_impl(
            field,
            self._state,
            self._state.spec,
            self.taper,
            self.truncation is not None,
        )


class SHTVectorRegrid(torch.nn.Module):
    """Fixed vector spectral regridding layer."""

    def __init__(
        self,
        source_grid: Grid,
        target_grid: Grid,
        truncation: str | TransformSpec | None = None,
        *,
        lmin: int | None = None,
        lmax: int | None = None,
        taper: float | None = None,
    ) -> None:
        super().__init__()
        _validate_grid(source_grid, "source_grid")
        _validate_grid(target_grid, "target_grid")
        selection = _resolve_spectral_spec(truncation, lmin=lmin, lmax=lmax)
        _validate_taper(taper)
        self.source_grid = source_grid
        self.target_grid = target_grid
        self.truncation = selection
        self.taper = taper
        self._state = _make_state(
            source_grid,
            target_grid,
            selection,
            vector=True,
            device=torch.device("cpu"),
        )
        _require_vector_bandwidth(self._state)

    def forward(self, u: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        _require_vector_tensors(u, v, self.source_grid)
        return _regrid_vector_impl(
            u,
            v,
            self._state,
            self._state.spec,
            self.taper,
            self.truncation is not None,
        )
