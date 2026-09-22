# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reusable analyzed representations for the PyTorch backend."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .._transform import TransformSpec
from ..grids import Grid
from ..spectral import _resolve_spectral_spec, _validate_taper
from ._backend import (
    _apply_selection,
    _check_selection,
    _intersect_transform_spec,
    _make_state,
    _require_tensor,
    _require_vector_bandwidth,
    _require_vector_tensors,
    _resolve_transform_spec,
    _TorchTransform,
    _validate_grid,
    _validate_radius,
)


def analyze(
    field: Tensor,
    truncation: str | TransformSpec | None = None,
    *,
    grid: Grid,
    lmin: int | None = None,
    lmax: int | None = None,
) -> SpectralField:
    """Analyze a tensor into a reusable scalar spectral field."""
    _validate_grid(grid)
    _require_tensor(field, grid)
    selection = _resolve_spectral_spec(truncation, lmin=lmin, lmax=lmax)
    state = _make_state(
        grid,
        grid,
        selection,
        vector=False,
        device=field.device,
    )
    coefficients = state.scalar_analysis(field)
    if selection is not None:
        coefficients = _apply_selection(coefficients, state, selection, None)
    return SpectralField(coefficients, state)


def analyze_vector(
    u: Tensor,
    v: Tensor,
    truncation: str | TransformSpec | None = None,
    *,
    grid: Grid,
    lmin: int | None = None,
    lmax: int | None = None,
) -> SpectralVectorField:
    """Analyze a geographic tensor vector into reusable E/B coefficients."""
    _validate_grid(grid)
    _require_vector_tensors(u, v, grid)
    selection = _resolve_spectral_spec(truncation, lmin=lmin, lmax=lmax)
    state = _make_state(
        grid,
        grid,
        selection,
        vector=True,
        device=u.device,
    )
    _require_vector_bandwidth(state)
    coefficients = state.vector_analysis(u, v)
    if selection is not None:
        coefficients = _apply_selection(coefficients, state, selection, None)
    return SpectralVectorField(coefficients, state)


@dataclass(frozen=True, slots=True)
class SpectralField:
    """An analyzed scalar field with private torch-harmonics coefficients."""

    _coefficients: Tensor
    _state: _TorchTransform

    @property
    def grid(self) -> Grid:
        """The source grid used for analysis."""
        grid = self._state.source_grid
        assert isinstance(grid, Grid)
        return grid

    @property
    def spec(self) -> TransformSpec:
        """The analyzed triangular coefficient domain."""
        spec = self._state.spec
        assert isinstance(spec, TransformSpec)
        return spec

    def filter(
        self,
        truncation: str | TransformSpec | None = None,
        *,
        lmin: int | None = None,
        lmax: int | None = None,
        taper: float | None = None,
    ) -> SpectralField:
        """Apply a spectral selection without synthesizing the field."""
        selection = _resolve_spectral_spec(truncation, lmin=lmin, lmax=lmax)
        _validate_taper(taper)
        effective = self._state.spec if selection is None else selection
        _check_selection(effective, self._state)
        if taper is None and (selection is None or selection == self.spec):
            return self
        return SpectralField(
            _apply_selection(self._coefficients, self._state, effective, taper),
            self._state,
        )

    def laplacian(self, *, radius: float = 6_371_220.0) -> SpectralField:
        """Apply the scalar spherical Laplacian to coefficients."""
        _validate_radius(radius)
        return SpectralField(
            self._coefficients
            * _laplacian_multiplier(self._state, radius, self._coefficients),
            self._state,
        )

    def inverse_laplacian(self, *, radius: float = 6_371_220.0) -> SpectralField:
        """Apply the zero-mode-defined inverse Laplacian to coefficients."""
        _validate_radius(radius)
        return SpectralField(
            self._coefficients
            * _inverse_laplacian_multiplier(self._state, radius, self._coefficients),
            self._state,
        )

    def synthesize(self) -> Tensor:
        """Synthesize the representation on its source grid."""
        return self._state.scalar_synthesis(self._coefficients)

    def regrid(
        self,
        target_grid: Grid,
        truncation: str | TransformSpec | None = None,
        *,
        lmin: int | None = None,
        lmax: int | None = None,
        taper: float | None = None,
    ) -> Tensor:
        """Synthesize the representation on another supported grid."""
        _validate_grid(target_grid, "target_grid")
        selection = _resolve_spectral_spec(truncation, lmin=lmin, lmax=lmax)
        _validate_taper(taper)
        effective = self._state.spec if selection is None else selection
        _check_selection(effective, self._state)
        if selection is None:
            target_spec = _intersect_transform_spec(
                self.grid, target_grid, self._state.spec
            )
        else:
            target_spec = _resolve_transform_spec(self.grid, target_grid, selection)
        state = _make_state(
            self.grid,
            target_grid,
            target_spec,
            vector=False,
            device=self._coefficients.device,
        )
        coefficients = self._coefficients
        if selection is None and taper is not None:
            coefficients = _apply_selection(
                coefficients, self._state, self._state.spec, taper
            )
        coefficients = _resize_coefficients(coefficients, state.spec)
        if selection is not None and (taper is not None or effective != self.spec):
            coefficients = _apply_selection(coefficients, state, effective, taper)
        return state.scalar_synthesis(coefficients)


@dataclass(frozen=True, slots=True)
class SpectralVectorField:
    """An analyzed geographic vector with private torch-harmonics coefficients."""

    _coefficients: Tensor
    _state: _TorchTransform

    @property
    def grid(self) -> Grid:
        """The source grid used for analysis."""
        grid = self._state.source_grid
        assert isinstance(grid, Grid)
        return grid

    @property
    def spec(self) -> TransformSpec:
        """The analyzed triangular coefficient domain."""
        spec = self._state.spec
        assert isinstance(spec, TransformSpec)
        return spec

    def filter(
        self,
        truncation: str | TransformSpec | None = None,
        *,
        lmin: int | None = None,
        lmax: int | None = None,
        taper: float | None = None,
    ) -> SpectralVectorField:
        """Apply a spectral selection without synthesizing the vector."""
        selection = _resolve_spectral_spec(truncation, lmin=lmin, lmax=lmax)
        _validate_taper(taper)
        effective = self._state.spec if selection is None else selection
        _check_selection(effective, self._state)
        if taper is None and (selection is None or selection == self.spec):
            return self
        return SpectralVectorField(
            _apply_selection(self._coefficients, self._state, effective, taper),
            self._state,
        )

    def laplacian(self, *, radius: float = 6_371_220.0) -> SpectralVectorField:
        """Apply the vector spherical Laplacian to E/B coefficients."""
        return self._laplacian(radius=radius, inverse=False)

    def inverse_laplacian(self, *, radius: float = 6_371_220.0) -> SpectralVectorField:
        """Apply the zero-mode-defined inverse vector Laplacian."""
        return self._laplacian(radius=radius, inverse=True)

    def vorticity(self, *, radius: float = 6_371_220.0) -> SpectralField:
        """Derive relative-vorticity coefficients from the vector."""
        _validate_radius(radius)
        scale = _degree_scale(self._state, radius, self._coefficients)
        coefficients = -self._coefficients.select(-3, 1) * scale
        return SpectralField(coefficients, self._state)

    def divergence(self, *, radius: float = 6_371_220.0) -> SpectralField:
        """Derive divergence coefficients from the vector."""
        _validate_radius(radius)
        scale = _degree_scale(self._state, radius, self._coefficients)
        coefficients = -self._coefficients.select(-3, 0) * scale
        return SpectralField(coefficients, self._state)

    def streamfunction(self, *, radius: float = 6_371_220.0) -> SpectralField:
        """Derive streamfunction coefficients from the vector."""
        return self._potential(radius, component=1)

    def velocity_potential(self, *, radius: float = 6_371_220.0) -> SpectralField:
        """Derive velocity-potential coefficients from the vector."""
        return self._potential(radius, component=0)

    def divergent(self) -> SpectralVectorField:
        """Keep only the divergent E component."""
        zeros = torch.zeros_like(self._coefficients.select(-3, 1))
        return SpectralVectorField(
            torch.stack((self._coefficients.select(-3, 0), zeros), dim=-3),
            self._state,
        )

    def rotational(self) -> SpectralVectorField:
        """Keep only the rotational B component."""
        zeros = torch.zeros_like(self._coefficients.select(-3, 0))
        return SpectralVectorField(
            torch.stack((zeros, self._coefficients.select(-3, 1)), dim=-3),
            self._state,
        )

    def synthesize(self) -> tuple[Tensor, Tensor]:
        """Synthesize eastward and northward components on the source grid."""
        return self._state.vector_synthesis(self._coefficients)

    def regrid(
        self,
        target_grid: Grid,
        truncation: str | TransformSpec | None = None,
        *,
        lmin: int | None = None,
        lmax: int | None = None,
        taper: float | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Synthesize the vector representation on another supported grid."""
        _validate_grid(target_grid, "target_grid")
        selection = _resolve_spectral_spec(truncation, lmin=lmin, lmax=lmax)
        _validate_taper(taper)
        effective = self._state.spec if selection is None else selection
        _check_selection(effective, self._state)
        if selection is None:
            target_spec = _intersect_transform_spec(
                self.grid, target_grid, self._state.spec
            )
        else:
            target_spec = _resolve_transform_spec(self.grid, target_grid, selection)
        state = _make_state(
            self.grid,
            target_grid,
            target_spec,
            vector=True,
            device=self._coefficients.device,
        )
        _require_vector_bandwidth(state)
        coefficients = self._coefficients
        if selection is None and taper is not None:
            coefficients = _apply_selection(
                coefficients, self._state, self._state.spec, taper
            )
        coefficients = _resize_coefficients(coefficients, state.spec)
        if selection is not None and (taper is not None or effective != self.spec):
            coefficients = _apply_selection(coefficients, state, effective, taper)
        return state.vector_synthesis(coefficients)

    def _laplacian(self, *, radius: float, inverse: bool) -> SpectralVectorField:
        _validate_radius(radius)
        multiplier = (
            _inverse_laplacian_multiplier(self._state, radius, self._coefficients)
            if inverse
            else _laplacian_multiplier(self._state, radius, self._coefficients)
        )
        return SpectralVectorField(self._coefficients * multiplier, self._state)

    def _potential(self, radius: float, *, component: int) -> SpectralField:
        _validate_radius(radius)
        scale = _degree_scale(self._state, radius, self._coefficients)
        inverse = _inverse_laplacian_multiplier(self._state, radius, self._coefficients)
        coefficients = -self._coefficients.select(-3, component) * scale * inverse
        return SpectralField(coefficients, self._state)


def _resize_coefficients(coefficients: Tensor, spec: TransformSpec) -> Tensor:
    """Resize torch-harmonics' rectangular triangular coefficient array."""
    target_l = spec.lmax + 1
    target_m = spec.mmax + 1
    if coefficients.shape[-2:] == (target_l, target_m):
        return coefficients
    source_l, source_m = coefficients.shape[-2:]
    result = coefficients[..., : min(source_l, target_l), : min(source_m, target_m)]
    if result.shape[-1] < target_m:
        result = torch.nn.functional.pad(result, (0, target_m - result.shape[-1]))
    if result.shape[-2] < target_l:
        result = torch.nn.functional.pad(result, (0, 0, 0, target_l - result.shape[-2]))
    return result


def _degree_scale(
    state: _TorchTransform, radius: float, coefficients: Tensor
) -> Tensor:
    return (
        state._degree_scale.to(
            device=coefficients.device, dtype=coefficients.real.dtype
        )
        / radius
    )


def _laplacian_multiplier(
    state: _TorchTransform, radius: float, coefficients: Tensor
) -> Tensor:
    degrees = state._degrees_tensor.to(
        device=coefficients.device, dtype=coefficients.real.dtype
    )
    return -(degrees * (degrees + 1.0)) / radius**2


def _inverse_laplacian_multiplier(
    state: _TorchTransform, radius: float, coefficients: Tensor
) -> Tensor:
    degrees = state._degrees_tensor.to(
        device=coefficients.device, dtype=coefficients.real.dtype
    )
    denominator = degrees * (degrees + 1.0)
    positive = denominator > 0.0
    safe = torch.where(positive, denominator, torch.ones_like(denominator))
    return torch.where(
        positive,
        -(radius**2) / safe,
        torch.zeros_like(denominator),
    )


__all__ = ["SpectralField", "SpectralVectorField", "analyze", "analyze_vector"]
