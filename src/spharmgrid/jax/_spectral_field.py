# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reusable analyzed representations for the JAX backend."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._transform import TransformSpec
from ..grids import Grid, clenshaw_curtis_grid, gaussian_grid, grid_layout
from ..spectral import (
    _intersect_transform_spec,
    _resolve_spectral_spec,
    _spectral_selection_is_within,
    _validate_taper,
)
from ._backend import (
    _apply_selection,
    _degree_scale,
    _inverse_laplacian_multiplier,
    _JaxTransform,
    _laplacian_multiplier,
    _make_transform,
    _require_array,
    _require_vector_arrays,
    _require_vector_bandwidth,
    _scalar_analysis,
    _scalar_synthesis,
    _validate_grid,
    _validate_radius,
    _vector_analysis,
    _vector_synthesis,
)


def analyze(
    field: Array,
    truncation: str | TransformSpec | None = None,
    *,
    grid: Grid,
    lmin: int | None = None,
    lmax: int | None = None,
) -> SpectralField:
    """Analyze a JAX array into a reusable scalar spectral field."""
    _validate_grid(grid)
    _require_array(field, grid)
    selection = _resolve_spectral_spec(truncation, lmin=lmin, lmax=lmax)
    transform = _make_transform(grid, grid, selection)
    coefficients = _scalar_analysis(field, transform)
    if selection is not None:
        coefficients = _apply_selection(coefficients, transform, None)
    return SpectralField(coefficients, transform)


def analyze_vector(
    u: Array,
    v: Array,
    truncation: str | TransformSpec | None = None,
    *,
    grid: Grid,
    lmin: int | None = None,
    lmax: int | None = None,
) -> SpectralVectorField:
    """Analyze a geographic JAX vector into reusable E/B coefficients."""
    _validate_grid(grid)
    _require_vector_arrays(u, v, grid)
    selection = _resolve_spectral_spec(truncation, lmin=lmin, lmax=lmax)
    transform = _make_transform(grid, grid, selection)
    _require_vector_bandwidth(transform)
    coefficients = _vector_analysis(u, v, transform)
    if selection is not None:
        coefficients = _apply_selection(coefficients, transform, None)
    return SpectralVectorField(coefficients, transform)


def _grid_key(grid: Grid) -> tuple[Any, ...]:
    layout = grid_layout(grid)
    latitude_order = (
        "ascending" if grid.latitude[0] < grid.latitude[-1] else "descending"
    )
    return (
        grid.kind,
        grid.nlat,
        grid.nlon,
        latitude_order,
        float(layout.longitude.canonical_values[0]),
        _longitude_order_key(layout.longitude.restore_indices),
    )


def _longitude_order_key(indices: np.ndarray) -> tuple[Any, ...]:
    """Encode common cyclic orders compactly while preserving arbitrary ones."""
    indices = np.asarray(indices, dtype=np.intp)
    if indices.size == 0:
        return ("identity",)
    shift = (-int(indices[0])) % indices.size
    expected = (np.arange(indices.size, dtype=np.intp) - shift) % indices.size
    if np.array_equal(indices, expected):
        return ("identity",) if shift == 0 else ("roll", shift)
    return ("indices", tuple(int(index) for index in indices))


def _grid_from_key(key: tuple[Any, ...]) -> Grid:
    kind, nlat, nlon, latitude_order, longitude_origin, longitude_order = key
    if kind == "gl":
        generated = gaussian_grid(
            nlat,
            nlon,
            lon0=longitude_origin,
            latitude_order=latitude_order,
        )
    else:
        generated = clenshaw_curtis_grid(
            nlat,
            nlon,
            lon0=longitude_origin,
            latitude_order=latitude_order,
        )
    order_kind = longitude_order[0]
    identity = np.arange(nlon, dtype=np.intp)
    if order_kind == "identity":
        restore_indices = identity
    elif order_kind == "roll":
        restore_indices = np.roll(identity, int(longitude_order[1]))
    else:
        restore_indices = np.asarray(longitude_order[1], dtype=np.intp)
    longitude = generated.longitude[restore_indices]
    return Grid(kind, generated.latitude, longitude)


def _transform_key(transform: _JaxTransform) -> tuple[Any, ...]:
    spec = transform.spec
    return (
        _grid_key(transform.source),
        _grid_key(transform.target),
        spec.lmin,
        spec.lmax,
        spec.mmax,
        spec.truncation,
    )


def _transform_from_key(key: tuple[Any, ...]) -> _JaxTransform:
    source_key, target_key, lmin, lmax, mmax, truncation = key
    source = _grid_from_key(source_key)
    target = _grid_from_key(target_key)
    spec = TransformSpec(lmin, lmax, mmax, truncation)
    return _make_transform(source, target, spec)


class SpectralField:
    """An analyzed scalar field with private S2FFT coefficients."""

    __slots__ = ("_coefficients", "_transform")

    def __init__(self, coefficients: Array, transform: _JaxTransform) -> None:
        self._coefficients = coefficients
        self._transform = transform

    @property
    def grid(self) -> Grid:
        """The source grid used for analysis."""
        return self._transform.source

    @property
    def spec(self) -> TransformSpec:
        """The spectral domain currently available in this object."""
        return self._transform.spec

    def tree_flatten(self) -> tuple[tuple[Array], tuple[Any, ...]]:
        return (self._coefficients,), (_transform_key(self._transform),)

    @classmethod
    def tree_unflatten(
        cls, aux_data: tuple[Any, ...], children: tuple[Array, ...]
    ) -> SpectralField:
        return cls(children[0], _transform_from_key(aux_data[0]))

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
        effective = self.spec if selection is None else selection
        _validate_selection(self.spec, effective)
        if taper is None and (selection is None or selection == self.spec):
            return self
        transform = _make_transform(self.grid, self.grid, effective)
        return SpectralField(
            _apply_selection(self._coefficients, transform, taper), transform
        )

    def laplacian(self, *, radius: float = 6_371_220.0) -> SpectralField:
        """Apply the scalar spherical Laplacian to coefficients."""
        _validate_radius(radius)
        multiplier = _laplacian_multiplier(
            self._transform.source_bandlimit,
            radius,
            jnp.real(self._coefficients).dtype,
        )
        return SpectralField(self._coefficients * multiplier, self._transform)

    def inverse_laplacian(self, *, radius: float = 6_371_220.0) -> SpectralField:
        """Apply the zero-mode-defined inverse Laplacian to coefficients."""
        _validate_radius(radius)
        multiplier = _inverse_laplacian_multiplier(
            self._transform.source_bandlimit,
            radius,
            jnp.real(self._coefficients).dtype,
        )
        return SpectralField(self._coefficients * multiplier, self._transform)

    def synthesize(self) -> Array:
        """Synthesize the representation on its source grid."""
        return _scalar_synthesis(self._coefficients, self._transform).astype(
            jnp.real(self._coefficients).dtype
        )

    def regrid(
        self,
        target_grid: Grid,
        truncation: str | TransformSpec | None = None,
        *,
        lmin: int | None = None,
        lmax: int | None = None,
        taper: float | None = None,
    ) -> Array:
        """Synthesize the representation on another supported grid."""
        _validate_grid(target_grid, "target_grid")
        selection = _resolve_spectral_spec(truncation, lmin=lmin, lmax=lmax)
        _validate_taper(taper)
        effective = self.spec if selection is None else selection
        _validate_selection(self.spec, effective)
        if selection is None:
            target_spec = _intersect_transform_spec(self.grid, target_grid, self.spec)
        else:
            target_spec = selection
        transform = _make_transform(self.grid, target_grid, target_spec)
        coefficients = self._coefficients
        if selection is None:
            if taper is not None:
                coefficients = _apply_selection(coefficients, self._transform, taper)
            if target_spec != self.spec:
                coefficients = _apply_selection(coefficients, transform, None)
        elif taper is not None or effective != self.spec:
            coefficients = _apply_selection(coefficients, transform, taper)
        return _scalar_synthesis(coefficients, transform).astype(
            jnp.real(self._coefficients).dtype
        )


class SpectralVectorField:
    """An analyzed geographic vector with private S2FFT coefficients."""

    __slots__ = ("_coefficients", "_transform")

    def __init__(self, coefficients: Array, transform: _JaxTransform) -> None:
        self._coefficients = coefficients
        self._transform = transform

    @property
    def grid(self) -> Grid:
        """The source grid used for analysis."""
        return self._transform.source

    @property
    def spec(self) -> TransformSpec:
        """The spectral domain currently available in this object."""
        return self._transform.spec

    def tree_flatten(self) -> tuple[tuple[Array], tuple[Any, ...]]:
        return (self._coefficients,), (_transform_key(self._transform),)

    @classmethod
    def tree_unflatten(
        cls, aux_data: tuple[Any, ...], children: tuple[Array, ...]
    ) -> SpectralVectorField:
        return cls(children[0], _transform_from_key(aux_data[0]))

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
        effective = self.spec if selection is None else selection
        _validate_selection(self.spec, effective)
        if taper is None and (selection is None or selection == self.spec):
            return self
        transform = _make_transform(self.grid, self.grid, effective)
        return SpectralVectorField(
            _apply_selection(self._coefficients, transform, taper), transform
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
        scale = (
            _degree_scale(
                self._transform.source_bandlimit,
                jnp.real(self._coefficients).dtype,
            )
            / radius
        )
        return SpectralField(
            -self._coefficients[..., 1, :, :] * scale,
            self._transform,
        )

    def divergence(self, *, radius: float = 6_371_220.0) -> SpectralField:
        """Derive divergence coefficients from the vector."""
        _validate_radius(radius)
        scale = (
            _degree_scale(
                self._transform.source_bandlimit,
                jnp.real(self._coefficients).dtype,
            )
            / radius
        )
        return SpectralField(
            -self._coefficients[..., 0, :, :] * scale,
            self._transform,
        )

    def streamfunction(self, *, radius: float = 6_371_220.0) -> SpectralField:
        """Derive streamfunction coefficients from the vector."""
        return self._potential(radius, component=1)

    def velocity_potential(self, *, radius: float = 6_371_220.0) -> SpectralField:
        """Derive velocity-potential coefficients from the vector."""
        return self._potential(radius, component=0)

    def divergent(self) -> SpectralVectorField:
        """Keep only the divergent E component."""
        zeros = jnp.zeros_like(self._coefficients[..., 1, :, :])
        return SpectralVectorField(
            jnp.stack((self._coefficients[..., 0, :, :], zeros), axis=-3),
            self._transform,
        )

    def rotational(self) -> SpectralVectorField:
        """Keep only the rotational B component."""
        zeros = jnp.zeros_like(self._coefficients[..., 0, :, :])
        return SpectralVectorField(
            jnp.stack((zeros, self._coefficients[..., 1, :, :]), axis=-3),
            self._transform,
        )

    def synthesize(self) -> tuple[Array, Array]:
        """Synthesize eastward and northward components on the source grid."""
        return _vector_synthesis(self._coefficients, self._transform)

    def regrid(
        self,
        target_grid: Grid,
        truncation: str | TransformSpec | None = None,
        *,
        lmin: int | None = None,
        lmax: int | None = None,
        taper: float | None = None,
    ) -> tuple[Array, Array]:
        """Synthesize the vector representation on another supported grid."""
        _validate_grid(target_grid, "target_grid")
        selection = _resolve_spectral_spec(truncation, lmin=lmin, lmax=lmax)
        _validate_taper(taper)
        effective = self.spec if selection is None else selection
        _validate_selection(self.spec, effective)
        if selection is None:
            target_spec = _intersect_transform_spec(self.grid, target_grid, self.spec)
        else:
            target_spec = selection
        transform = _make_transform(self.grid, target_grid, target_spec)
        _require_vector_bandwidth(transform)
        coefficients = self._coefficients
        if selection is None:
            if taper is not None:
                coefficients = _apply_selection(coefficients, self._transform, taper)
            if target_spec != self.spec:
                coefficients = _apply_selection(coefficients, transform, None)
        elif taper is not None or effective != self.spec:
            coefficients = _apply_selection(coefficients, transform, taper)
        return _vector_synthesis(coefficients, transform)

    def _laplacian(self, *, radius: float, inverse: bool) -> SpectralVectorField:
        _validate_radius(radius)
        multiplier = (
            _inverse_laplacian_multiplier(
                self._transform.source_bandlimit,
                radius,
                jnp.real(self._coefficients).dtype,
            )
            if inverse
            else _laplacian_multiplier(
                self._transform.source_bandlimit,
                radius,
                jnp.real(self._coefficients).dtype,
            )
        )
        return SpectralVectorField(self._coefficients * multiplier, self._transform)

    def _potential(self, radius: float, *, component: int) -> SpectralField:
        _validate_radius(radius)
        scale = (
            _degree_scale(
                self._transform.source_bandlimit,
                jnp.real(self._coefficients).dtype,
            )
            / radius
        )
        inverse = _inverse_laplacian_multiplier(
            self._transform.source_bandlimit,
            radius,
            jnp.real(self._coefficients).dtype,
        )
        coefficients = -self._coefficients[..., component, :, :] * scale * inverse
        return SpectralField(coefficients, self._transform)


def _validate_selection(
    analyzed: TransformSpec, selection: TransformSpec | None
) -> None:
    if not _spectral_selection_is_within(analyzed, selection):
        raise ValueError(
            f"requested spectral selection {selection} exceeds the current "
            f"spectral domain {analyzed}; discarded modes cannot be restored"
        )


jax.tree_util.register_pytree_node_class(SpectralField)
jax.tree_util.register_pytree_node_class(SpectralVectorField)


__all__ = ["SpectralField", "SpectralVectorField", "analyze", "analyze_vector"]
