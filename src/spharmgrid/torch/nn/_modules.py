# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reusable ``torch.nn.Module`` wrappers for spharmgrid Torch operations."""

from __future__ import annotations

from typing import cast

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
    _resolve_transform_spec,
    _TorchTransform,
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

    The object keeps only the transform directions required by the most recent
    operation. Repeated operations with the same direction requirements reuse
    that state; switching operation families replaces it. Move the module to the
    input tensor's device and dtype with ``.to(device=..., dtype=...)``.
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
        _resolve_transform_spec(grid, grid, None)
        self.grid = grid
        self.radius = radius
        self._active_directions: tuple[bool, bool, bool, bool] | None = None
        self._active_transform: _TorchTransform | None = None
        self.register_buffer(
            "_anchor",
            torch.empty(0, dtype=torch.float64),
            persistent=False,
        )

    def _transform_for(
        self,
        *,
        scalar_analysis: bool = False,
        scalar_synthesis: bool = False,
        vector_analysis: bool = False,
        vector_synthesis: bool = False,
    ) -> _TorchTransform:
        directions = (
            scalar_analysis,
            scalar_synthesis,
            vector_analysis,
            vector_synthesis,
        )
        if self._active_transform is None or self._active_directions != directions:
            anchor = cast(Tensor, self._anchor)
            self._active_transform = _make_state(
                self.grid,
                self.grid,
                None,
                scalar_analysis=scalar_analysis,
                scalar_synthesis=scalar_synthesis,
                vector_analysis=vector_analysis,
                vector_synthesis=vector_synthesis,
                device=anchor.device,
                dtype=anchor.dtype,
            )
            self._active_directions = directions
        active = self._active_transform
        assert isinstance(active, _TorchTransform)
        return active

    def filter(
        self,
        field: Tensor,
        truncation: str | TransformSpec | None = None,
        *,
        lmin: int | None = None,
        lmax: int | None = None,
        taper: float | None = None,
    ) -> Tensor:
        """Apply a hard or Sardeshmukh--Hoskins spectral selection."""
        selection = _resolve_spectral_spec(truncation, lmin=lmin, lmax=lmax)
        _validate_taper(taper)
        _require_tensor(field, self.grid)
        state = self._transform_for(
            scalar_analysis=True,
            scalar_synthesis=True,
        )
        return _filter_impl(
            field,
            state,
            selection or state.spec,
            taper,
        )

    def gradient(self, field: Tensor) -> tuple[Tensor, Tensor]:
        """Return the physical eastward and northward gradient."""
        _require_tensor(field, self.grid)
        state = self._transform_for(
            scalar_analysis=True,
            vector_synthesis=True,
        )
        return _gradient_with_state(field, state, self.radius)

    def inverse_gradient(self, eastward: Tensor, northward: Tensor) -> Tensor:
        """Recover the irrotational scalar potential from a vector field."""
        _require_vector_tensors(eastward, northward, self.grid)
        state = self._transform_for(
            scalar_synthesis=True,
            vector_analysis=True,
        )
        return _inverse_gradient_with_state(
            eastward,
            northward,
            state,
            self.radius,
        )

    def laplacian(self, field: Tensor) -> Tensor:
        """Apply the physical scalar spherical Laplacian."""
        _require_tensor(field, self.grid)
        state = self._transform_for(
            scalar_analysis=True,
            scalar_synthesis=True,
        )
        return _laplacian_with_state(field, state, self.radius)

    def inverse_laplacian(self, field: Tensor) -> Tensor:
        """Solve the scalar inverse Laplacian with a zero degree-zero mode."""
        _require_tensor(field, self.grid)
        state = self._transform_for(
            scalar_analysis=True,
            scalar_synthesis=True,
        )
        return _inverse_laplacian_with_state(field, state, self.radius)

    def vector_laplacian(self, u: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Apply the vector spherical Laplacian to geographic wind."""
        _require_vector_tensors(u, v, self.grid)
        state = self._transform_for(
            vector_analysis=True,
            vector_synthesis=True,
        )
        return _vector_laplacian_with_state(
            u,
            v,
            state,
            self.radius,
            False,
        )

    def inverse_vector_laplacian(
        self,
        u: Tensor,
        v: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Solve the vector inverse Laplacian with degree zero removed."""
        _require_vector_tensors(u, v, self.grid)
        state = self._transform_for(
            vector_analysis=True,
            vector_synthesis=True,
        )
        return _vector_laplacian_with_state(
            u,
            v,
            state,
            self.radius,
            True,
        )

    def vorticity(self, u: Tensor, v: Tensor) -> Tensor:
        """Compute relative vorticity from eastward and northward wind."""
        _require_vector_tensors(u, v, self.grid)
        state = self._transform_for(
            scalar_synthesis=True,
            vector_analysis=True,
        )
        return _kinematics_with_state(u, v, state, self.radius)[0]

    def divergence(self, u: Tensor, v: Tensor) -> Tensor:
        """Compute horizontal wind divergence."""
        _require_vector_tensors(u, v, self.grid)
        state = self._transform_for(
            scalar_synthesis=True,
            vector_analysis=True,
        )
        return _kinematics_with_state(u, v, state, self.radius)[1]

    def kinematics(self, u: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(vorticity, divergence)`` from one vector analysis."""
        _require_vector_tensors(u, v, self.grid)
        state = self._transform_for(
            scalar_synthesis=True,
            vector_analysis=True,
        )
        return _kinematics_with_state(u, v, state, self.radius)

    def streamfunction(self, u: Tensor, v: Tensor) -> Tensor:
        """Compute streamfunction from a wind field."""
        _require_vector_tensors(u, v, self.grid)
        state = self._transform_for(
            scalar_synthesis=True,
            vector_analysis=True,
        )
        return _potentials_with_state(u, v, state, self.radius)[0]

    def velocity_potential(self, u: Tensor, v: Tensor) -> Tensor:
        """Compute velocity potential from a wind field."""
        _require_vector_tensors(u, v, self.grid)
        state = self._transform_for(
            scalar_synthesis=True,
            vector_analysis=True,
        )
        return _potentials_with_state(u, v, state, self.radius)[1]

    def potentials(self, u: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(streamfunction, velocity_potential)`` from one analysis."""
        _require_vector_tensors(u, v, self.grid)
        state = self._transform_for(
            scalar_synthesis=True,
            vector_analysis=True,
        )
        return _potentials_with_state(u, v, state, self.radius)

    def helmholtz(self, u: Tensor, v: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return divergent and rotational eastward and northward wind."""
        _require_vector_tensors(u, v, self.grid)
        state = self._transform_for(
            vector_analysis=True,
            vector_synthesis=True,
        )
        return _helmholtz_with_state(u, v, state)

    def rotational_wind(
        self,
        field: Tensor,
        *,
        source: RotationalWindSource,
    ) -> tuple[Tensor, Tensor]:
        """Recover rotational wind from vorticity or streamfunction."""
        _validate_scalar_source(source, ("vorticity", "streamfunction"))
        _require_tensor(field, self.grid)
        state = self._transform_for(
            scalar_analysis=True,
            vector_synthesis=True,
        )
        return _single_source_wind_with_state(
            field,
            state,
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
        """Recover divergent wind from divergence or velocity potential."""
        _validate_scalar_source(
            source,
            ("divergence", "velocity_potential"),
        )
        _require_tensor(field, self.grid)
        state = self._transform_for(
            scalar_analysis=True,
            vector_synthesis=True,
        )
        return _single_source_wind_with_state(
            field,
            state,
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
        """Reconstruct wind from vorticity/divergence or the two potentials."""
        _validate_source(source)
        _require_vector_tensors(first, second, self.grid)
        state = self._transform_for(
            scalar_analysis=True,
            vector_synthesis=True,
        )
        return _wind_with_state(first, second, state, source, self.radius)


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
            scalar_analysis=True,
            scalar_synthesis=True,
            device=torch.device("cpu"),
        )

    def forward(self, field: Tensor) -> Tensor:
        """Apply the configured spectral filter."""
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
            scalar_analysis=True,
            scalar_synthesis=True,
            device=torch.device("cpu"),
        )

    def forward(self, field: Tensor) -> Tensor:
        """Regrid a scalar tensor to the configured target grid."""
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
            vector_analysis=True,
            vector_synthesis=True,
            device=torch.device("cpu"),
        )
        _require_vector_bandwidth(self._state)

    def forward(self, u: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Regrid eastward and northward wind to the target grid."""
        _require_vector_tensors(u, v, self.source_grid)
        return _regrid_vector_impl(
            u,
            v,
            self._state,
            self._state.spec,
            self.taper,
            self.truncation is not None,
        )
