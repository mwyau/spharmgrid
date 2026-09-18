"""Tensor-native spherical harmonic operations for the Torch backend."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from .._kinematics_types import (
    DivergentWindSource,
    RotationalWindSource,
    WindSource,
)
from .._transform import TransformSpec
from ..grids import Grid
from ..operators import EARTH_RADIUS_M
from ..spectral import _resolve_spectral_spec, _validate_taper
from ._backend import (
    _apply_selection,
    _check_selection,
    _make_state,
    _require_tensor,
    _require_vector_bandwidth,
    _require_vector_tensors,
    _TorchTransform,
    _validate_grid,
    _validate_radius,
)


def filter(
    field: Tensor,
    truncation: str | TransformSpec | None = None,
    *,
    grid: Grid,
    lmin: int | None = None,
    lmax: int | None = None,
    taper: float | None = None,
) -> Tensor:
    """Apply a hard or Sardeshmukh--Hoskins spectral selection."""
    _validate_grid(grid)
    _require_tensor(field, grid)
    selection = _resolve_spectral_spec(truncation, lmin=lmin, lmax=lmax)
    _validate_taper(taper)
    state = _make_state(
        grid,
        grid,
        selection,
        vector=False,
        device=field.device,
    )
    return _filter_impl(field, state, state.spec, taper)


def regrid(
    field: Tensor,
    target_grid: Grid,
    truncation: str | TransformSpec | None = None,
    *,
    source_grid: Grid,
    lmin: int | None = None,
    lmax: int | None = None,
    taper: float | None = None,
) -> Tensor:
    """Spectrally regrid a scalar tensor between two supported grids."""
    _validate_grid(source_grid, "source_grid")
    _validate_grid(target_grid, "target_grid")
    _require_tensor(field, source_grid)
    selection = _resolve_spectral_spec(truncation, lmin=lmin, lmax=lmax)
    _validate_taper(taper)
    state = _make_state(
        source_grid,
        target_grid,
        selection,
        vector=False,
        device=field.device,
    )
    return _regrid_impl(field, state, state.spec, taper, selection is not None)


def regrid_vector(
    u: Tensor,
    v: Tensor,
    target_grid: Grid,
    truncation: str | TransformSpec | None = None,
    *,
    source_grid: Grid,
    lmin: int | None = None,
    lmax: int | None = None,
    taper: float | None = None,
) -> tuple[Tensor, Tensor]:
    """Regrid eastward and northward wind in one vector transform cycle."""
    _validate_grid(source_grid, "source_grid")
    _validate_grid(target_grid, "target_grid")
    _require_vector_tensors(u, v, source_grid)
    selection = _resolve_spectral_spec(truncation, lmin=lmin, lmax=lmax)
    _validate_taper(taper)
    state = _make_state(
        source_grid,
        target_grid,
        selection,
        vector=True,
        device=u.device,
    )
    _require_vector_bandwidth(state)
    return _regrid_vector_impl(
        u,
        v,
        state,
        state.spec,
        taper,
        selection is not None,
    )


def gradient(
    field: Tensor,
    *,
    grid: Grid,
    radius: float = EARTH_RADIUS_M,
) -> tuple[Tensor, Tensor]:
    """Return the physical eastward and northward gradient."""
    _validate_grid(grid)
    _require_tensor(field, grid)
    _validate_radius(radius)
    state = _full_state(field, grid, vector=True)
    return _gradient_with_state(field, state, radius)


def _gradient_with_state(
    field: Tensor,
    state: _TorchTransform,
    radius: float,
) -> tuple[Tensor, Tensor]:
    scalar_coefficients = state.scalar_analysis(field)
    scale = state._degree_scale.to(device=field.device, dtype=field.dtype) / radius
    vector_coefficients = torch.stack(
        (scalar_coefficients * scale, torch.zeros_like(scalar_coefficients)),
        dim=-3,
    )
    return state.vector_synthesis(vector_coefficients)


def inverse_gradient(
    eastward: Tensor,
    northward: Tensor,
    *,
    grid: Grid,
    radius: float = EARTH_RADIUS_M,
) -> Tensor:
    """Recover the irrotational scalar potential from a vector field."""
    _validate_grid(grid)
    _require_vector_tensors(eastward, northward, grid)
    _validate_radius(radius)
    state = _full_state(eastward, grid, vector=True)
    return _inverse_gradient_with_state(eastward, northward, state, radius)


def _inverse_gradient_with_state(
    eastward: Tensor,
    northward: Tensor,
    state: _TorchTransform,
    radius: float,
) -> Tensor:
    _require_vector_bandwidth(state)
    vector_coefficients = state.vector_analysis(eastward, northward)
    scale = (
        state._degree_scale.to(device=eastward.device, dtype=eastward.dtype) / radius
    )
    positive = scale > 0.0
    safe_scale = torch.where(positive, scale, torch.ones_like(scale))
    potential = torch.where(
        positive,
        vector_coefficients.select(-3, 0) / safe_scale,
        torch.zeros_like(vector_coefficients.select(-3, 0)),
    )
    return state.scalar_synthesis(potential)


def laplacian(
    field: Tensor,
    *,
    grid: Grid,
    radius: float = EARTH_RADIUS_M,
) -> Tensor:
    """Apply the physical scalar spherical Laplacian."""
    _validate_grid(grid)
    _require_tensor(field, grid)
    _validate_radius(radius)
    state = _full_state(field, grid, vector=False)
    return _laplacian_with_state(field, state, radius)


def _laplacian_with_state(
    field: Tensor,
    state: _TorchTransform,
    radius: float,
) -> Tensor:
    coefficients = state.scalar_analysis(field)
    multiplier = _laplacian_multiplier(state, radius, field.dtype)
    return state.scalar_synthesis(coefficients * multiplier)


def inverse_laplacian(
    field: Tensor,
    *,
    grid: Grid,
    radius: float = EARTH_RADIUS_M,
) -> Tensor:
    """Solve the scalar inverse Laplacian with a zero degree-zero mode."""
    _validate_grid(grid)
    _require_tensor(field, grid)
    _validate_radius(radius)
    state = _full_state(field, grid, vector=False)
    return _inverse_laplacian_with_state(field, state, radius)


def _inverse_laplacian_with_state(
    field: Tensor,
    state: _TorchTransform,
    radius: float,
) -> Tensor:
    coefficients = state.scalar_analysis(field)
    multiplier = _inverse_laplacian_multiplier(state, radius, field.dtype)
    return state.scalar_synthesis(coefficients * multiplier)


def vector_laplacian(
    u: Tensor,
    v: Tensor,
    *,
    grid: Grid,
    radius: float = EARTH_RADIUS_M,
) -> tuple[Tensor, Tensor]:
    """Apply the vector spherical Laplacian to geographic wind."""
    return _vector_laplacian(u, v, grid=grid, radius=radius, inverse=False)


def inverse_vector_laplacian(
    u: Tensor,
    v: Tensor,
    *,
    grid: Grid,
    radius: float = EARTH_RADIUS_M,
) -> tuple[Tensor, Tensor]:
    """Solve the vector inverse Laplacian with degree zero removed."""
    return _vector_laplacian(u, v, grid=grid, radius=radius, inverse=True)


def vorticity(
    u: Tensor,
    v: Tensor,
    *,
    grid: Grid,
    radius: float = EARTH_RADIUS_M,
) -> Tensor:
    """Compute relative vorticity from eastward and northward wind."""
    vo, _ = _kinematics_impl(u, v, grid=grid, radius=radius)
    return vo


def divergence(
    u: Tensor,
    v: Tensor,
    *,
    grid: Grid,
    radius: float = EARTH_RADIUS_M,
) -> Tensor:
    """Compute horizontal wind divergence."""
    _, div = _kinematics_impl(u, v, grid=grid, radius=radius)
    return div


def kinematics(
    u: Tensor,
    v: Tensor,
    *,
    grid: Grid,
    radius: float = EARTH_RADIUS_M,
) -> tuple[Tensor, Tensor]:
    """Return ``(vorticity, divergence)`` from one vector analysis."""
    return _kinematics_impl(u, v, grid=grid, radius=radius)


def streamfunction(
    u: Tensor,
    v: Tensor,
    *,
    grid: Grid,
    radius: float = EARTH_RADIUS_M,
) -> Tensor:
    """Compute streamfunction from a wind field."""
    psi, _ = _potentials_impl(u, v, grid=grid, radius=radius)
    return psi


def velocity_potential(
    u: Tensor,
    v: Tensor,
    *,
    grid: Grid,
    radius: float = EARTH_RADIUS_M,
) -> Tensor:
    """Compute velocity potential from a wind field."""
    _, chi = _potentials_impl(u, v, grid=grid, radius=radius)
    return chi


def potentials(
    u: Tensor,
    v: Tensor,
    *,
    grid: Grid,
    radius: float = EARTH_RADIUS_M,
) -> tuple[Tensor, Tensor]:
    """Return ``(streamfunction, velocity_potential)`` from one analysis."""
    return _potentials_impl(u, v, grid=grid, radius=radius)


def helmholtz(
    u: Tensor,
    v: Tensor,
    *,
    grid: Grid,
    radius: float = EARTH_RADIUS_M,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return divergent eastward/northward and rotational eastward/northward wind."""
    _validate_grid(grid)
    _require_vector_tensors(u, v, grid)
    _validate_radius(radius)
    state = _full_state(u, grid, vector=True)
    return _helmholtz_with_state(u, v, state)


def _helmholtz_with_state(
    u: Tensor,
    v: Tensor,
    state: _TorchTransform,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    _require_vector_bandwidth(state)
    coefficients = state.vector_analysis(u, v)
    zeros = torch.zeros_like(coefficients.select(-3, 0))
    divergent = state.vector_synthesis(
        torch.stack((coefficients.select(-3, 0), zeros), dim=-3)
    )
    rotational = state.vector_synthesis(
        torch.stack((zeros, coefficients.select(-3, 1)), dim=-3)
    )
    return (*divergent, *rotational)


def rotational_wind(
    field: Tensor,
    *,
    grid: Grid,
    source: RotationalWindSource,
    radius: float = EARTH_RADIUS_M,
) -> tuple[Tensor, Tensor]:
    """Recover rotational wind from vorticity or streamfunction."""
    _validate_scalar_source(source, ("vorticity", "streamfunction"))
    return _single_source_wind(
        field,
        grid=grid,
        source=source,
        kind="rotational",
        radius=radius,
    )


def divergent_wind(
    field: Tensor,
    *,
    grid: Grid,
    source: DivergentWindSource,
    radius: float = EARTH_RADIUS_M,
) -> tuple[Tensor, Tensor]:
    """Recover divergent wind from divergence or velocity potential."""
    _validate_scalar_source(
        source,
        ("divergence", "velocity_potential"),
    )
    return _single_source_wind(
        field,
        grid=grid,
        source=source,
        kind="divergent",
        radius=radius,
    )


def wind(
    first: Tensor,
    second: Tensor,
    *,
    grid: Grid,
    source: WindSource,
    radius: float = EARTH_RADIUS_M,
) -> tuple[Tensor, Tensor]:
    """Reconstruct wind from vorticity/divergence or the two potentials."""
    _validate_source(source)
    _validate_grid(grid)
    _require_vector_tensors(first, second, grid)
    _validate_radius(radius)
    state = _full_state(first, grid, vector=True)
    return _wind_with_state(first, second, state, source, radius)


def _wind_with_state(
    first: Tensor,
    second: Tensor,
    state: _TorchTransform,
    source: str,
    radius: float,
) -> tuple[Tensor, Tensor]:
    _require_vector_bandwidth(state)
    first_coefficients = state.scalar_analysis(first)
    second_coefficients = state.scalar_analysis(second)
    if source == "vorticity_divergence":
        vo = first_coefficients
        div = second_coefficients
    else:
        multiplier = _laplacian_multiplier(state, radius, first.dtype)
        vo = first_coefficients * multiplier
        div = second_coefficients * multiplier
    scale = state._degree_scale.to(device=first.device, dtype=first.dtype) / radius
    e = _safe_divide(-div, scale)
    b = _safe_divide(-vo, scale)
    return state.vector_synthesis(torch.stack((e, b), dim=-3))


def _filter_impl(
    field: Tensor,
    state: _TorchTransform,
    selection: TransformSpec,
    taper: float | None,
) -> Tensor:
    _check_selection(selection, state)
    coefficients = state.scalar_analysis(field)
    selected = _apply_selection(coefficients, state, selection, taper)
    return state.scalar_synthesis(selected)


def _regrid_impl(
    field: Tensor,
    state: _TorchTransform,
    selection: TransformSpec,
    taper: float | None,
    apply_selection: bool,
) -> Tensor:
    coefficients = state.scalar_analysis(field)
    if apply_selection or taper is not None:
        coefficients = _apply_selection(coefficients, state, selection, taper)
    return state.scalar_synthesis(coefficients)


def _regrid_vector_impl(
    u: Tensor,
    v: Tensor,
    state: _TorchTransform,
    selection: TransformSpec,
    taper: float | None,
    apply_selection: bool,
) -> tuple[Tensor, Tensor]:
    coefficients = state.vector_analysis(u, v)
    if apply_selection or taper is not None:
        coefficients = _apply_selection(coefficients, state, selection, taper)
    return state.vector_synthesis(coefficients)


def _full_state(field: Tensor, grid: Grid, *, vector: bool) -> _TorchTransform:
    return _make_state(
        grid,
        grid,
        None,
        vector=vector,
        device=field.device,
    )


def _vector_laplacian(
    u: Tensor,
    v: Tensor,
    *,
    grid: Grid,
    radius: float,
    inverse: bool,
) -> tuple[Tensor, Tensor]:
    _validate_grid(grid)
    _require_vector_tensors(u, v, grid)
    _validate_radius(radius)
    state = _full_state(u, grid, vector=True)
    return _vector_laplacian_with_state(u, v, state, radius, inverse)


def _vector_laplacian_with_state(
    u: Tensor,
    v: Tensor,
    state: _TorchTransform,
    radius: float,
    inverse: bool,
) -> tuple[Tensor, Tensor]:
    _require_vector_bandwidth(state)
    coefficients = state.vector_analysis(u, v)
    multiplier = (
        _inverse_laplacian_multiplier(state, radius, u.dtype)
        if inverse
        else _laplacian_multiplier(state, radius, u.dtype)
    )
    return state.vector_synthesis(coefficients * multiplier)


def _kinematics_impl(
    u: Tensor,
    v: Tensor,
    *,
    grid: Grid,
    radius: float,
) -> tuple[Tensor, Tensor]:
    _validate_grid(grid)
    _require_vector_tensors(u, v, grid)
    _validate_radius(radius)
    state = _full_state(u, grid, vector=True)
    return _kinematics_with_state(u, v, state, radius)


def _kinematics_with_state(
    u: Tensor,
    v: Tensor,
    state: _TorchTransform,
    radius: float,
) -> tuple[Tensor, Tensor]:
    _require_vector_bandwidth(state)
    coefficients = state.vector_analysis(u, v)
    scale = state._degree_scale.to(device=u.device, dtype=u.dtype) / radius
    scalar_coefficients = torch.stack(
        (-coefficients.select(-3, 1) * scale, -coefficients.select(-3, 0) * scale),
        dim=-3,
    )
    maps = state.scalar_synthesis(scalar_coefficients)
    return maps.select(-3, 0), maps.select(-3, 1)


def _potentials_impl(
    u: Tensor,
    v: Tensor,
    *,
    grid: Grid,
    radius: float,
) -> tuple[Tensor, Tensor]:
    _validate_grid(grid)
    _require_vector_tensors(u, v, grid)
    _validate_radius(radius)
    state = _full_state(u, grid, vector=True)
    return _potentials_with_state(u, v, state, radius)


def _potentials_with_state(
    u: Tensor,
    v: Tensor,
    state: _TorchTransform,
    radius: float,
) -> tuple[Tensor, Tensor]:
    _require_vector_bandwidth(state)
    coefficients = state.vector_analysis(u, v)
    scale = state._degree_scale.to(device=u.device, dtype=u.dtype) / radius
    vo = -coefficients.select(-3, 1) * scale
    div = -coefficients.select(-3, 0) * scale
    inverse = _inverse_laplacian_multiplier(state, radius, u.dtype)
    scalar_coefficients = torch.stack((vo * inverse, div * inverse), dim=-3)
    maps = state.scalar_synthesis(scalar_coefficients)
    return maps.select(-3, 0), maps.select(-3, 1)


def _single_source_wind(
    field: Tensor,
    *,
    grid: Grid,
    source: str,
    kind: Literal["rotational", "divergent"],
    radius: float,
) -> tuple[Tensor, Tensor]:
    _validate_grid(grid)
    _require_tensor(field, grid)
    _validate_radius(radius)
    state = _full_state(field, grid, vector=True)
    return _single_source_wind_with_state(field, state, source, kind, radius)


def _single_source_wind_with_state(
    field: Tensor,
    state: _TorchTransform,
    source: str,
    kind: Literal["rotational", "divergent"],
    radius: float,
) -> tuple[Tensor, Tensor]:
    _require_vector_bandwidth(state)
    scalar_coefficients = state.scalar_analysis(field)
    scale = state._degree_scale.to(device=field.device, dtype=field.dtype) / radius
    if kind == "rotational":
        source_coefficients = (
            scalar_coefficients
            if source == "vorticity"
            else scalar_coefficients * _laplacian_multiplier(state, radius, field.dtype)
        )
        e = torch.zeros_like(source_coefficients)
        b = _safe_divide(-source_coefficients, scale)
    else:
        source_coefficients = (
            scalar_coefficients
            if source == "divergence"
            else scalar_coefficients * _laplacian_multiplier(state, radius, field.dtype)
        )
        e = _safe_divide(-source_coefficients, scale)
        b = torch.zeros_like(source_coefficients)
    return state.vector_synthesis(torch.stack((e, b), dim=-3))


def _laplacian_multiplier(
    state: _TorchTransform,
    radius: float,
    dtype: torch.dtype,
) -> Tensor:
    degrees = state._degrees_tensor.to(dtype=dtype)
    return -(degrees * (degrees + 1.0)) / radius**2


def _inverse_laplacian_multiplier(
    state: _TorchTransform,
    radius: float,
    dtype: torch.dtype,
) -> Tensor:
    degrees = state._degrees_tensor.to(dtype=dtype)
    denominator = degrees * (degrees + 1.0)
    positive = denominator > 0.0
    safe_denominator = torch.where(
        positive,
        denominator,
        torch.ones_like(denominator),
    )
    return torch.where(
        positive,
        -(radius**2) / safe_denominator,
        torch.zeros_like(denominator),
    )


def _safe_divide(numerator: Tensor, denominator: Tensor) -> Tensor:
    positive = denominator > 0.0
    safe_denominator = torch.where(
        positive,
        denominator,
        torch.ones_like(denominator),
    )
    return torch.where(
        positive,
        numerator / safe_denominator,
        torch.zeros_like(numerator),
    )


def _validate_scalar_source(
    source: str | None,
    allowed: tuple[str, str],
) -> None:
    if source not in allowed:
        raise ValueError(f"source must be one of: {', '.join(allowed)}")


def _validate_source(source: str | None) -> None:
    if source not in ("vorticity_divergence", "potentials"):
        raise ValueError("source must be 'vorticity_divergence' or 'potentials'")
