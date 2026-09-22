# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""JAX execution helpers backed by S2FFT."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import cast

import jax
import jax.numpy as jnp
import s2fft
from jax import Array, core
from jax.typing import DTypeLike

from .._transform import TransformSpec
from ..grids import Grid, GridLayout, grid_layout
from ..spectral import _validate_taper, resolve_transform_spec


@dataclass(frozen=True, slots=True)
class _JaxTransform:
    """Static transform information for one source and target grid pair."""

    source: Grid
    target: Grid
    spec: TransformSpec
    source_layout: GridLayout
    target_layout: GridLayout
    source_bandlimit: int
    target_bandlimit: int


def _validate_grid(grid: Grid, name: str = "grid") -> None:
    if not isinstance(grid, Grid):
        raise TypeError(f"{name} must be a spharmgrid.Grid")
    bandlimit = _native_bandlimit(grid)
    expected_nlon = 2 * bandlimit - 1 if grid.kind == "gl" else 2 * bandlimit
    if grid.nlon != expected_nlon:
        sampling = "GL" if grid.kind == "gl" else "CC/MWSS"
        raise ValueError(
            f"spharmgrid.jax supports {sampling} grids only with shape "
            f"({bandlimit}, {expected_nlon}) for the S2FFT band-limit; got "
            f"({grid.nlat}, {grid.nlon})"
            if grid.kind == "gl"
            else f"spharmgrid.jax supports {sampling} grids only with shape "
            f"({bandlimit + 1}, {expected_nlon}) for the S2FFT band-limit; got "
            f"({grid.nlat}, {grid.nlon})"
        )


def _native_bandlimit(grid: Grid) -> int:
    return grid.nlat if grid.kind == "gl" else grid.nlat - 1


def _sampling(grid: Grid) -> str:
    return "gl" if grid.kind == "gl" else "mwss"


def _make_transform(
    source: Grid,
    target: Grid,
    selection: TransformSpec | None,
) -> _JaxTransform:
    _validate_grid(source, "source_grid")
    _validate_grid(target, "target_grid")
    spec = resolve_transform_spec(source, target, selection)
    return _JaxTransform(
        source=source,
        target=target,
        spec=spec,
        source_layout=grid_layout(source),
        target_layout=grid_layout(target),
        source_bandlimit=_native_bandlimit(source),
        target_bandlimit=_native_bandlimit(target),
    )


def _is_jax_array(value: object) -> bool:
    return isinstance(value, (jax.Array, core.Tracer))


def _require_array(field: Array, grid: Grid, name: str = "field") -> None:
    if not _is_jax_array(field):
        raise TypeError(f"{name} must be a jax.Array")
    if not bool(jax.config.read("jax_enable_x64")):
        raise RuntimeError(
            "spharmgrid.jax requires JAX x64 mode; set "
            "jax_enable_x64=True before creating arrays or using spharmgrid.jax. "
            "spharmgrid does not change this process-wide setting"
        )
    if field.ndim < 2:
        raise ValueError(f"{name} must have at least two dimensions")
    if tuple(field.shape[-2:]) != (grid.nlat, grid.nlon):
        raise ValueError(
            f"{name} must end with ({grid.nlat}, {grid.nlon}) spatial dimensions; "
            f"got {tuple(field.shape[-2:])}"
        )
    if field.dtype != jnp.float64:
        raise TypeError(
            f"{name} must use float64; float32 and complex64 inputs are not "
            "supported because the measured S2FFT transform paths are not "
            f"sufficiently accurate in single precision (got {field.dtype})"
        )


def _require_vector_arrays(u: Array, v: Array, grid: Grid) -> None:
    _require_array(u, grid, "u")
    _require_array(v, grid, "v")
    if u.shape != v.shape:
        raise ValueError(
            f"u and v must have the same shape; got {u.shape} and {v.shape}"
        )
    if u.dtype != v.dtype:
        raise TypeError(
            f"u and v must have the same dtype; got {u.dtype} and {v.dtype}"
        )


def _validate_radius(radius: float) -> None:
    if isinstance(radius, bool) or not isinstance(radius, (int, float)):
        raise TypeError("radius must be a positive finite number in metres")
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be a positive finite number in metres")


def _canonicalize(field: Array, layout: GridLayout) -> Array:
    latitude_indices = jnp.asarray(layout.latitude.canonical_indices, dtype=jnp.int32)
    longitude_indices = jnp.asarray(layout.longitude.canonical_indices, dtype=jnp.int32)
    result = jnp.take(field, latitude_indices, axis=-2)
    return jnp.take(result, longitude_indices, axis=-1)


def _restore(field: Array, layout: GridLayout) -> Array:
    latitude_indices = jnp.asarray(layout.latitude.restore_indices, dtype=jnp.int32)
    longitude_indices = jnp.asarray(layout.longitude.restore_indices, dtype=jnp.int32)
    result = jnp.take(field, latitude_indices, axis=-2)
    return jnp.take(result, longitude_indices, axis=-1)


def _complex_dtype(real_dtype: DTypeLike) -> DTypeLike:
    if real_dtype == jnp.float64 and bool(jax.config.read("jax_enable_x64")):
        return jnp.complex128
    return jnp.complex64


def _imaginary_unit(real_dtype: DTypeLike) -> Array:
    return jnp.asarray(1j, dtype=_complex_dtype(real_dtype))


def _phase(
    length: int, phi0_radians: float, dtype: DTypeLike, *, inverse: bool
) -> Array:
    modes = jnp.arange(-(length - 1), length, dtype=jnp.int32)
    sign = 1.0 if inverse else -1.0
    unit = jnp.asarray(1j, dtype=dtype)
    return jnp.exp(sign * unit * modes * phi0_radians)


@lru_cache(maxsize=32)
def _precomputes(
    bandlimit: int,
    sampling: str,
    spin: int,
    *,
    forward: bool,
) -> tuple[Array, ...]:
    """Memoize S2FFT's O(L²) recursion arrays for static transform settings."""
    with jax.ensure_compile_time_eval():
        values = tuple(
            cast(Array, value)
            for value in s2fft.generate_precomputes_jax(
                bandlimit,
                spin=spin,
                sampling=sampling,
                forward=forward,
            )
        )
        for value in values:
            value.block_until_ready()
    return values


def _forward_one(
    field: Array,
    bandlimit: int,
    sampling: str,
    spin: int,
) -> Array:
    return cast(
        Array,
        s2fft.forward_jax(
            field,
            bandlimit,
            spin=spin,
            sampling=sampling,
            reality=spin == 0,
            precomps=_precomputes(bandlimit, sampling, spin, forward=True),
            spmd=False,
        ),
    )


def _inverse_one(
    coefficients: Array,
    bandlimit: int,
    sampling: str,
    spin: int,
) -> Array:
    return cast(
        Array,
        s2fft.inverse_jax(
            coefficients,
            bandlimit,
            spin=spin,
            sampling=sampling,
            reality=spin == 0,
            precomps=_precomputes(bandlimit, sampling, spin, forward=False),
            spmd=False,
        ),
    )


def _forward(field: Array, bandlimit: int, sampling: str, spin: int) -> Array:
    leading_shape = field.shape[:-2]
    flattened = field.reshape((-1, field.shape[-2], field.shape[-1]))
    result = jax.vmap(lambda frame: _forward_one(frame, bandlimit, sampling, spin))(
        flattened
    )
    return result.reshape(leading_shape + (bandlimit, 2 * bandlimit - 1))


def _inverse(coefficients: Array, bandlimit: int, sampling: str, spin: int) -> Array:
    leading_shape = coefficients.shape[:-2]
    flattened = coefficients.reshape((-1, bandlimit, 2 * bandlimit - 1))
    result = jax.vmap(lambda frame: _inverse_one(frame, bandlimit, sampling, spin))(
        flattened
    )
    return result.reshape(leading_shape + result.shape[-2:])


def _scalar_analysis(field: Array, transform: _JaxTransform) -> Array:
    canonical = _canonicalize(field, transform.source_layout)
    coefficients = _forward(
        canonical,
        transform.source_bandlimit,
        _sampling(transform.source),
        0,
    )
    phase = _phase(
        transform.source_bandlimit,
        transform.source_layout.phi0_radians,
        coefficients.dtype,
        inverse=False,
    )
    return coefficients * phase


def _spin_analysis(field: Array, transform: _JaxTransform, spin: int) -> Array:
    canonical = _canonicalize(field, transform.source_layout)
    coefficients = _forward(
        canonical,
        transform.source_bandlimit,
        _sampling(transform.source),
        spin,
    )
    phase = _phase(
        transform.source_bandlimit,
        transform.source_layout.phi0_radians,
        coefficients.dtype,
        inverse=False,
    )
    return coefficients * phase


def _resize(
    coefficients: Array,
    source_bandlimit: int,
    target_bandlimit: int,
) -> Array:
    if source_bandlimit == target_bandlimit:
        return coefficients
    common = min(source_bandlimit, target_bandlimit)
    source_start = source_bandlimit - common
    target_start = target_bandlimit - common
    result = jnp.zeros(
        coefficients.shape[:-2] + (target_bandlimit, 2 * target_bandlimit - 1),
        dtype=coefficients.dtype,
    )
    source_slice = coefficients[
        ..., :common, source_start : source_start + 2 * common - 1
    ]
    return result.at[..., :common, target_start : target_start + 2 * common - 1].set(
        source_slice
    )


def _scalar_synthesis(coefficients: Array, transform: _JaxTransform) -> Array:
    resized = _resize(
        coefficients,
        transform.source_bandlimit,
        transform.target_bandlimit,
    )
    phase = _phase(
        transform.target_bandlimit,
        transform.target_layout.phi0_radians,
        resized.dtype,
        inverse=True,
    )
    values = _inverse(
        resized * phase,
        transform.target_bandlimit,
        _sampling(transform.target),
        0,
    )
    return _restore(jnp.real(values), transform.target_layout)


def _spin_synthesis(
    coefficients: Array,
    transform: _JaxTransform,
    spin: int,
) -> Array:
    resized = _resize(
        coefficients,
        transform.source_bandlimit,
        transform.target_bandlimit,
    )
    phase = _phase(
        transform.target_bandlimit,
        transform.target_layout.phi0_radians,
        resized.dtype,
        inverse=True,
    )
    values = _inverse(
        resized * phase,
        transform.target_bandlimit,
        _sampling(transform.target),
        spin,
    )
    return _restore(values, transform.target_layout)


def _spectral_weights(
    transform: _JaxTransform, taper: float | None, dtype: DTypeLike
) -> Array:
    _validate_taper(taper)
    spec = transform.spec
    degrees = jnp.arange(transform.source_bandlimit, dtype=jnp.int32)[:, None]
    orders = jnp.abs(
        jnp.arange(
            -(transform.source_bandlimit - 1),
            transform.source_bandlimit,
            dtype=jnp.int32,
        )
    )[None, :]
    inside = (degrees >= spec.lmin) & (degrees <= spec.lmax) & (orders <= spec.mmax)
    if spec.truncation == "rhomboidal":
        inside &= degrees - orders <= spec.lmax - spec.mmax
    if taper is None:
        response = jnp.ones_like(degrees, dtype=dtype)
    elif spec.lmax == 0:
        response = jnp.full_like(degrees, float(taper), dtype=dtype)
    else:
        degree_values = degrees.astype(dtype)
        coefficient = (
            -jnp.log(jnp.asarray(taper, dtype=degree_values.dtype))
            / (spec.lmax * (spec.lmax + 1)) ** 2
        )
        response = jnp.exp(-coefficient * (degree_values * (degree_values + 1.0)) ** 2)
    return jnp.where(inside, response, jnp.zeros_like(response))


def _apply_selection(
    coefficients: Array,
    transform: _JaxTransform,
    taper: float | None,
) -> Array:
    weights = _spectral_weights(transform, taper, jnp.real(coefficients).dtype)
    return coefficients * weights


def _degree_scale(bandlimit: int, dtype: DTypeLike) -> Array:
    degrees = jnp.arange(bandlimit, dtype=dtype)[:, None]
    return jnp.sqrt(degrees * (degrees + 1.0))


def _laplacian_multiplier(bandlimit: int, radius: float, dtype: DTypeLike) -> Array:
    degrees = jnp.arange(bandlimit, dtype=dtype)[:, None]
    return -(degrees * (degrees + 1.0)) / radius**2


def _inverse_laplacian_multiplier(
    bandlimit: int, radius: float, dtype: DTypeLike
) -> Array:
    degrees = jnp.arange(bandlimit, dtype=dtype)[:, None]
    denominator = degrees * (degrees + 1.0)
    positive = denominator > 0.0
    safe_denominator = jnp.where(positive, denominator, jnp.ones_like(denominator))
    return jnp.where(
        positive,
        -(radius**2) / safe_denominator,
        jnp.zeros_like(denominator),
    )


def _safe_divide(numerator: Array, denominator: Array) -> Array:
    positive = denominator > 0.0
    safe_denominator = jnp.where(positive, denominator, jnp.ones_like(denominator))
    return jnp.where(positive, numerator / safe_denominator, jnp.zeros_like(numerator))


def _vector_analysis(
    u: Array,
    v: Array,
    transform: _JaxTransform,
) -> Array:
    # Geographic components have polar components v_theta=-v and v_phi=u.
    # S2FFT's spin fields are q_+=v_theta+i*v_phi and q_-=v_theta-i*v_phi.
    # The DUCC-compatible vector coefficients are E=(a_- - a_+)/2 and
    # B=i*(a_+ + a_-)/2; this mapping fixes both the component order and signs.
    unit = _imaginary_unit(u.dtype)
    q_plus = -v + unit * u
    q_minus = -v - unit * u
    plus = _spin_analysis(q_plus, transform, 1)
    minus = _spin_analysis(q_minus, transform, -1)
    electric = 0.5 * (minus - plus)
    magnetic = 0.5j * (plus + minus)
    return jnp.stack((electric, magnetic), axis=-3)


def _vector_synthesis(
    coefficients: Array, transform: _JaxTransform
) -> tuple[Array, Array]:
    # Invert q_+=-E-i*B and q_-=E-i*B, then convert polar components back to
    # geographic eastward and northward components.
    unit = _imaginary_unit(jnp.real(coefficients).dtype)
    q_plus = -coefficients[..., 0, :, :] - unit * coefficients[..., 1, :, :]
    q_minus = coefficients[..., 0, :, :] - unit * coefficients[..., 1, :, :]
    plus = _spin_synthesis(q_plus, transform, 1)
    minus = _spin_synthesis(q_minus, transform, -1)
    v_theta = 0.5 * (plus + minus)
    v_phi = (plus - minus) / (2.0 * unit)
    return (
        jnp.real(v_phi).astype(jnp.real(coefficients).dtype),
        jnp.real(-v_theta).astype(jnp.real(coefficients).dtype),
    )


def _require_vector_bandwidth(transform: _JaxTransform) -> None:
    if transform.spec.lmax < 1:
        raise ValueError("vector operation requires a grid supporting total degree l=1")


def _validate_source(source: str | None) -> None:
    if source not in ("vorticity_divergence", "potentials"):
        raise ValueError("source must be 'vorticity_divergence' or 'potentials'")


def _validate_scalar_source(source: str | None, allowed: tuple[str, str]) -> None:
    if source not in allowed:
        raise ValueError(f"source must be one of: {', '.join(allowed)}")


def _resolve_spectral_spec(
    truncation: str | TransformSpec | None,
    *,
    lmin: int | None,
    lmax: int | None,
) -> TransformSpec | None:
    if truncation is not None and (lmin is not None or lmax is not None):
        raise ValueError("use either truncation= or explicit lmin= and lmax=, not both")
    if truncation is not None:
        if isinstance(truncation, TransformSpec):
            return truncation
        from ..spectral import parse_spectral

        return parse_spectral(truncation)
    if lmin is None and lmax is None:
        return None
    if lmin is None or lmax is None:
        raise ValueError("explicit spectral bounds require both lmin= and lmax=")
    return TransformSpec(lmin, lmax, lmax)


__all__ = [
    "_JaxTransform",
    "_apply_selection",
    "_degree_scale",
    "_inverse_laplacian_multiplier",
    "_laplacian_multiplier",
    "_make_transform",
    "_require_array",
    "_require_vector_arrays",
    "_require_vector_bandwidth",
    "_resolve_spectral_spec",
    "_safe_divide",
    "_scalar_analysis",
    "_scalar_synthesis",
    "_spin_analysis",
    "_spin_synthesis",
    "_validate_grid",
    "_validate_radius",
    "_validate_scalar_source",
    "_validate_source",
    "_vector_analysis",
    "_vector_synthesis",
]
