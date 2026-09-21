# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tensor-native spherical harmonic operations for the JAX backend."""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
from jax import Array

from .._kinematics_types import DivergentWindSource, RotationalWindSource, WindSource
from .._transform import TransformSpec
from ..grids import Grid
from ..operators import EARTH_RADIUS_M
from ..spectral import _validate_taper
from ._backend import (
    _apply_selection,
    _degree_scale,
    _inverse_laplacian_multiplier,
    _laplacian_multiplier,
    _make_transform,
    _require_array,
    _require_vector_arrays,
    _require_vector_bandwidth,
    _resolve_spectral_spec,
    _safe_divide,
    _scalar_analysis,
    _scalar_synthesis,
    _validate_grid,
    _validate_radius,
    _validate_scalar_source,
    _validate_source,
    _vector_analysis,
    _vector_synthesis,
)


def filter(
    field: Array,
    truncation: str | TransformSpec | None = None,
    *,
    grid: Grid,
    lmin: int | None = None,
    lmax: int | None = None,
    taper: float | None = None,
) -> Array:
    """Apply a hard or Sardeshmukh–Hoskins spectral selection."""
    _validate_grid(grid)
    _require_array(field, grid)
    selection = _resolve_spectral_spec(truncation, lmin=lmin, lmax=lmax)
    _validate_taper(taper)
    transform = _make_transform(grid, grid, selection)
    coefficients = _apply_selection(
        _scalar_analysis(field, transform), transform, taper
    )
    return _scalar_synthesis(coefficients, transform)


def regrid(
    field: Array,
    target_grid: Grid,
    truncation: str | TransformSpec | None = None,
    *,
    source_grid: Grid,
    lmin: int | None = None,
    lmax: int | None = None,
    taper: float | None = None,
) -> Array:
    """Spectrally regrid a scalar array between supported grids."""
    _validate_grid(source_grid, "source_grid")
    _validate_grid(target_grid, "target_grid")
    _require_array(field, source_grid)
    selection = _resolve_spectral_spec(truncation, lmin=lmin, lmax=lmax)
    _validate_taper(taper)
    transform = _make_transform(source_grid, target_grid, selection)
    coefficients = _apply_selection(
        _scalar_analysis(field, transform), transform, taper
    )
    return _scalar_synthesis(coefficients, transform)


def regrid_vector(
    u: Array,
    v: Array,
    target_grid: Grid,
    truncation: str | TransformSpec | None = None,
    *,
    source_grid: Grid,
    lmin: int | None = None,
    lmax: int | None = None,
    taper: float | None = None,
) -> tuple[Array, Array]:
    """Spectrally regrid eastward and northward wind components together."""
    _validate_grid(source_grid, "source_grid")
    _validate_grid(target_grid, "target_grid")
    _require_vector_arrays(u, v, source_grid)
    selection = _resolve_spectral_spec(truncation, lmin=lmin, lmax=lmax)
    _validate_taper(taper)
    transform = _make_transform(source_grid, target_grid, selection)
    _require_vector_bandwidth(transform)
    coefficients = _vector_analysis(u, v, transform)
    selected = _apply_selection(coefficients, transform, taper)
    output_u, output_v = _vector_synthesis(selected, transform)
    return output_u, output_v


def gradient(
    field: Array,
    *,
    grid: Grid,
    radius: float = EARTH_RADIUS_M,
) -> tuple[Array, Array]:
    """Return the physical eastward and northward gradient."""
    _validate_grid(grid)
    _require_array(field, grid)
    _validate_radius(radius)
    transform = _make_transform(grid, grid, None)
    scalar_coefficients = _scalar_analysis(field, transform)
    scale = _degree_scale(transform.source_bandlimit, field.dtype) / radius
    vector_coefficients = jnp.stack(
        (scalar_coefficients * scale, jnp.zeros_like(scalar_coefficients)), axis=-3
    )
    eastward, northward = _vector_synthesis(vector_coefficients, transform)
    return eastward, northward


def inverse_gradient(
    eastward: Array,
    northward: Array,
    *,
    grid: Grid,
    radius: float = EARTH_RADIUS_M,
) -> Array:
    """Recover the irrotational scalar potential from a vector field."""
    _validate_grid(grid)
    _require_vector_arrays(eastward, northward, grid)
    _validate_radius(radius)
    transform = _make_transform(grid, grid, None)
    _require_vector_bandwidth(transform)
    coefficients = _vector_analysis(eastward, northward, transform)
    scale = _degree_scale(transform.source_bandlimit, eastward.dtype) / radius
    potential = _safe_divide(coefficients[..., 0, :, :], scale)
    return _scalar_synthesis(potential, transform)


def laplacian(
    field: Array,
    *,
    grid: Grid,
    radius: float = EARTH_RADIUS_M,
) -> Array:
    """Apply the physical scalar spherical Laplacian."""
    _validate_grid(grid)
    _require_array(field, grid)
    _validate_radius(radius)
    transform = _make_transform(grid, grid, None)
    coefficients = _scalar_analysis(field, transform)
    multiplier = _laplacian_multiplier(transform.source_bandlimit, radius, field.dtype)
    return _scalar_synthesis(coefficients * multiplier, transform)


def inverse_laplacian(
    field: Array,
    *,
    grid: Grid,
    radius: float = EARTH_RADIUS_M,
) -> Array:
    """Solve the scalar inverse Laplacian with a zero degree-zero mode."""
    _validate_grid(grid)
    _require_array(field, grid)
    _validate_radius(radius)
    transform = _make_transform(grid, grid, None)
    coefficients = _scalar_analysis(field, transform)
    multiplier = _inverse_laplacian_multiplier(
        transform.source_bandlimit, radius, field.dtype
    )
    return _scalar_synthesis(coefficients * multiplier, transform)


def vector_laplacian(
    u: Array,
    v: Array,
    *,
    grid: Grid,
    radius: float = EARTH_RADIUS_M,
) -> tuple[Array, Array]:
    """Apply the vector spherical Laplacian to a geographic wind field."""
    return _vector_laplacian(u, v, grid=grid, radius=radius, inverse=False)


def inverse_vector_laplacian(
    u: Array,
    v: Array,
    *,
    grid: Grid,
    radius: float = EARTH_RADIUS_M,
) -> tuple[Array, Array]:
    """Solve the vector inverse Laplacian with degree zero removed."""
    return _vector_laplacian(u, v, grid=grid, radius=radius, inverse=True)


def vorticity(
    u: Array,
    v: Array,
    *,
    grid: Grid,
    radius: float = EARTH_RADIUS_M,
) -> Array:
    """Compute relative vorticity from eastward and northward wind."""
    result, _ = _kinematics_impl(u, v, grid=grid, radius=radius)
    return result


def divergence(
    u: Array,
    v: Array,
    *,
    grid: Grid,
    radius: float = EARTH_RADIUS_M,
) -> Array:
    """Compute horizontal wind divergence."""
    _, result = _kinematics_impl(u, v, grid=grid, radius=radius)
    return result


def kinematics(
    u: Array,
    v: Array,
    *,
    grid: Grid,
    radius: float = EARTH_RADIUS_M,
) -> tuple[Array, Array]:
    """Return ``(vorticity, divergence)`` from one vector analysis."""
    return _kinematics_impl(u, v, grid=grid, radius=radius)


def streamfunction(
    u: Array,
    v: Array,
    *,
    grid: Grid,
    radius: float = EARTH_RADIUS_M,
) -> Array:
    """Compute streamfunction from an eastward and northward wind field."""
    result, _ = _potentials_impl(u, v, grid=grid, radius=radius)
    return result


def velocity_potential(
    u: Array,
    v: Array,
    *,
    grid: Grid,
    radius: float = EARTH_RADIUS_M,
) -> Array:
    """Compute velocity potential from an eastward and northward wind field."""
    _, result = _potentials_impl(u, v, grid=grid, radius=radius)
    return result


def potentials(
    u: Array,
    v: Array,
    *,
    grid: Grid,
    radius: float = EARTH_RADIUS_M,
) -> tuple[Array, Array]:
    """Return ``(streamfunction, velocity_potential)`` from one analysis."""
    return _potentials_impl(u, v, grid=grid, radius=radius)


def helmholtz(
    u: Array,
    v: Array,
    *,
    grid: Grid,
    radius: float = EARTH_RADIUS_M,
) -> tuple[Array, Array, Array, Array]:
    """Return divergent and rotational eastward/northward wind components."""
    _validate_grid(grid)
    _require_vector_arrays(u, v, grid)
    _validate_radius(radius)
    transform = _make_transform(grid, grid, None)
    _require_vector_bandwidth(transform)
    coefficients = _vector_analysis(u, v, transform)
    zeros = jnp.zeros_like(coefficients[..., 0, :, :])
    divergent = _vector_synthesis(
        jnp.stack((coefficients[..., 0, :, :], zeros), axis=-3), transform
    )
    rotational = _vector_synthesis(
        jnp.stack((zeros, coefficients[..., 1, :, :]), axis=-3), transform
    )
    return (*divergent, *rotational)


def rotational_wind(
    field: Array,
    *,
    grid: Grid,
    source: RotationalWindSource,
    radius: float = EARTH_RADIUS_M,
) -> tuple[Array, Array]:
    """Recover rotational wind from vorticity or streamfunction."""
    return _single_source_wind(
        field,
        grid=grid,
        source=source,
        kind="rotational",
        radius=radius,
    )


def divergent_wind(
    field: Array,
    *,
    grid: Grid,
    source: DivergentWindSource,
    radius: float = EARTH_RADIUS_M,
) -> tuple[Array, Array]:
    """Recover divergent wind from divergence or velocity potential."""
    return _single_source_wind(
        field,
        grid=grid,
        source=source,
        kind="divergent",
        radius=radius,
    )


def wind(
    first: Array,
    second: Array,
    *,
    grid: Grid,
    source: WindSource,
    radius: float = EARTH_RADIUS_M,
) -> tuple[Array, Array]:
    """Reconstruct wind from vorticity/divergence or the two potentials."""
    _validate_source(source)
    _validate_grid(grid)
    _require_array(first, grid, "first")
    _require_array(second, grid, "second")
    if first.shape != second.shape:
        raise ValueError(
            f"first and second must have the same shape; got {first.shape} and "
            f"{second.shape}"
        )
    if first.dtype != second.dtype:
        raise TypeError(
            f"first and second must have the same dtype; got {first.dtype} and "
            f"{second.dtype}"
        )
    _validate_radius(radius)
    transform = _make_transform(grid, grid, None)
    _require_vector_bandwidth(transform)
    first_coefficients = _scalar_analysis(first, transform)
    second_coefficients = _scalar_analysis(second, transform)
    if source == "vorticity_divergence":
        vorticity_coefficients, divergence_coefficients = (
            first_coefficients,
            second_coefficients,
        )
    else:
        multiplier = _laplacian_multiplier(
            transform.source_bandlimit, radius, first.dtype
        )
        vorticity_coefficients = first_coefficients * multiplier
        divergence_coefficients = second_coefficients * multiplier
    scale = _degree_scale(transform.source_bandlimit, first.dtype) / radius
    e = _safe_divide(-divergence_coefficients, scale)
    b = _safe_divide(-vorticity_coefficients, scale)
    return _vector_synthesis(jnp.stack((e, b), axis=-3), transform)


def _vector_laplacian(
    u: Array,
    v: Array,
    *,
    grid: Grid,
    radius: float,
    inverse: bool,
) -> tuple[Array, Array]:
    _validate_grid(grid)
    _require_vector_arrays(u, v, grid)
    _validate_radius(radius)
    transform = _make_transform(grid, grid, None)
    _require_vector_bandwidth(transform)
    coefficients = _vector_analysis(u, v, transform)
    multiplier = (
        _inverse_laplacian_multiplier(transform.source_bandlimit, radius, u.dtype)
        if inverse
        else _laplacian_multiplier(transform.source_bandlimit, radius, u.dtype)
    )
    return _vector_synthesis(coefficients * multiplier, transform)


def _kinematics_impl(
    u: Array,
    v: Array,
    *,
    grid: Grid,
    radius: float,
) -> tuple[Array, Array]:
    _validate_grid(grid)
    _require_vector_arrays(u, v, grid)
    _validate_radius(radius)
    transform = _make_transform(grid, grid, None)
    _require_vector_bandwidth(transform)
    coefficients = _vector_analysis(u, v, transform)
    scale = _degree_scale(transform.source_bandlimit, u.dtype) / radius
    scalar_coefficients = jnp.stack(
        (
            -coefficients[..., 1, :, :] * scale,
            -coefficients[..., 0, :, :] * scale,
        ),
        axis=-3,
    )
    maps = _scalar_synthesis(scalar_coefficients, transform)
    return maps[..., 0, :, :], maps[..., 1, :, :]


def _potentials_impl(
    u: Array,
    v: Array,
    *,
    grid: Grid,
    radius: float,
) -> tuple[Array, Array]:
    _validate_grid(grid)
    _require_vector_arrays(u, v, grid)
    _validate_radius(radius)
    transform = _make_transform(grid, grid, None)
    _require_vector_bandwidth(transform)
    coefficients = _vector_analysis(u, v, transform)
    scale = _degree_scale(transform.source_bandlimit, u.dtype) / radius
    vo = -coefficients[..., 1, :, :] * scale
    div = -coefficients[..., 0, :, :] * scale
    inverse = _inverse_laplacian_multiplier(transform.source_bandlimit, radius, u.dtype)
    maps = _scalar_synthesis(
        jnp.stack((vo * inverse, div * inverse), axis=-3), transform
    )
    return maps[..., 0, :, :], maps[..., 1, :, :]


def _single_source_wind(
    field: Array,
    *,
    grid: Grid,
    source: Literal["vorticity", "streamfunction", "divergence", "velocity_potential"],
    kind: Literal["rotational", "divergent"],
    radius: float,
) -> tuple[Array, Array]:
    _validate_grid(grid)
    _require_array(field, grid)
    _validate_radius(radius)
    allowed = (
        ("vorticity", "streamfunction")
        if kind == "rotational"
        else (
            "divergence",
            "velocity_potential",
        )
    )
    _validate_scalar_source(source, allowed)
    transform = _make_transform(grid, grid, None)
    _require_vector_bandwidth(transform)
    scalar_coefficients = _scalar_analysis(field, transform)
    if source in ("streamfunction", "velocity_potential"):
        scalar_coefficients = scalar_coefficients * _laplacian_multiplier(
            transform.source_bandlimit, radius, field.dtype
        )
    scale = _degree_scale(transform.source_bandlimit, field.dtype) / radius
    if kind == "rotational":
        e = jnp.zeros_like(scalar_coefficients)
        b = _safe_divide(-scalar_coefficients, scale)
    else:
        e = _safe_divide(-scalar_coefficients, scale)
        b = jnp.zeros_like(scalar_coefficients)
    return _vector_synthesis(jnp.stack((e, b), axis=-3), transform)
