# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scalar spherical harmonic filtering primitives built on DUCC."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import cast

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ._ducc import (
    alm_degrees,
    alm_orders,
    geometry_for,
    resolve_sht_threads,
    scalar_analysis,
    scalar_synthesis,
)
from ._transform import TransformSpec
from ._xarray import (
    FieldLayout,
    apply_ufunc_options,
    field_layout,
    require_dataarray,
    restore_output,
)
from .grids import Grid, GridCapabilities, grid_capabilities
from .metadata import preserve_quantity_metadata

_TRIANGULAR_RE = re.compile(r"^t(\d+)$", re.IGNORECASE)
_TRIANGULAR_BAND_RE = re.compile(r"^t(\d+)-(\d+)$", re.IGNORECASE)
_TRAPEZOIDAL_RE = re.compile(r"^t(\d+)x(\d+)$", re.IGNORECASE)
_RHOMBOIDAL_RE = re.compile(r"^r(\d+)$", re.IGNORECASE)


def parse_spectral(value: str) -> TransformSpec:
    """Parse case-insensitive ``Tn``, ``Tn-m``, ``Tnxm``, or ``Rn`` notation.

    An en dash is accepted in place of the ASCII hyphen.  The notation is a
    spectral selection, not a grid name.
    """
    if not isinstance(value, str):
        raise TypeError("spectral notation must be a string")
    normalized = value.strip().replace("–", "-")
    match = _TRIANGULAR_RE.fullmatch(normalized)
    if match is not None:
        limit = int(match.group(1))
        return TransformSpec(0, limit, limit)

    match = _TRIANGULAR_BAND_RE.fullmatch(normalized)
    if match is not None:
        return TransformSpec(
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(2)),
        )

    match = _TRAPEZOIDAL_RE.fullmatch(normalized)
    if match is not None:
        return TransformSpec(
            0,
            int(match.group(1)),
            int(match.group(2)),
            "trapezoidal",
        )

    match = _RHOMBOIDAL_RE.fullmatch(normalized)
    if match is not None:
        limit = int(match.group(1))
        return TransformSpec(0, 2 * limit, limit, "rhomboidal")

    raise ValueError("spectral notation must be Tn, Ta-b, Tnxm, or Rn, for example T42")


def filter(
    field: xr.DataArray,
    truncation: str | TransformSpec | None = None,
    *,
    lmin: int | None = None,
    lmax: int | None = None,
    taper: float | None = None,
    sht_threads: int | None = None,
) -> xr.DataArray:
    """Apply a hard or Sardeshmukh–Hoskins tapered spectral selection.

    With ``taper=None`` (the default), all modes inside the selected inclusive
    range are unchanged and all other modes are zero.  ``taper`` is the
    response at the upper retained total wavenumber.
    """
    field = require_dataarray(field)
    source = field_layout(field)
    selection = _resolve_spectral_spec(truncation, lmin=lmin, lmax=lmax)
    spec = resolve_transform_spec(source.grid, source.grid, selection)
    _validate_taper(taper)
    nthreads = resolve_sht_threads(sht_threads, dask=field.chunks is not None)

    def transform(frame: NDArray[np.generic]) -> NDArray[np.float64]:
        alm = scalar_analysis(
            frame,
            spec=spec,
            geometry=geometry_for(source.grid),
            phi0=source.transform_layout.phi0_radians,
            nthreads=nthreads,
        )
        filtered = apply_spectral_selection(alm, spec, taper)
        return scalar_synthesis(
            filtered,
            spec=spec,
            geometry=geometry_for(source.grid),
            ntheta=source.grid.nlat,
            nphi=source.grid.nlon,
            phi0=source.transform_layout.phi0_radians,
            nthreads=nthreads,
        )

    result = scalar_transform(field, source, source, transform)
    return preserve_quantity_metadata(result, field)


def resolve_transform_spec(
    source: Grid,
    target: Grid,
    selection: TransformSpec | None,
) -> TransformSpec:
    """Choose a DUCC bandwidth without silently clamping explicit selection input."""
    source_capabilities = grid_capabilities(source)
    target_capabilities = grid_capabilities(target)
    if selection is not None:
        if selection.truncation == "triangular":
            supported = min(
                source_capabilities.triangular_lmax,
                target_capabilities.triangular_lmax,
            )
            if selection.lmax > supported:
                raise ValueError(
                    f"requested lmax={selection.lmax} exceeds the supported triangular "
                    f"bandwidth {supported} for the source and target grids"
                )
        else:
            _validate_nontriangular_capabilities(
                selection,
                source_capabilities,
                target_capabilities,
            )
        return selection

    # Without a T-range the source and target can retain high-degree zonally
    # symmetric content when latitude sampling permits it, while m remains
    # bounded by both longitude samplings.
    lmax = min(source_capabilities.latitude_lmax, target_capabilities.latitude_lmax)
    mmax = min(
        lmax,
        source_capabilities.longitude_mmax,
        target_capabilities.longitude_mmax,
    )
    truncation = "triangular" if lmax == mmax else "trapezoidal"
    return TransformSpec(0, lmax, mmax, truncation)


def _intersect_transform_spec(
    source: Grid,
    target: Grid,
    analyzed: TransformSpec,
) -> TransformSpec:
    """Resolve an analyzed domain without exceeding target capabilities."""
    try:
        return resolve_transform_spec(source, target, analyzed)
    except ValueError:
        available = resolve_transform_spec(source, target, None)
        lmax = min(analyzed.lmax, available.lmax)
        mmax = min(analyzed.mmax, available.mmax)
        lmin = min(analyzed.lmin, lmax)
        truncation = (
            "triangular"
            if analyzed.truncation == "triangular" and lmax == mmax
            else "trapezoidal"
        )
        return TransformSpec(lmin, lmax, mmax, truncation)


def apply_spectral_selection(
    alm: NDArray[np.complexfloating],
    spec: TransformSpec,
    taper: float | None,
) -> NDArray[np.complex128]:
    """Return coefficients masked to a retained spectral domain."""
    weights = _spectral_selection_weights(spec, spec, taper)
    result = np.array(alm, dtype=np.complex128, copy=True)
    weight_shape = (1,) * (result.ndim - 1) + (weights.size,)
    result *= weights.reshape(weight_shape)
    return result


def _spectral_selection_weights(
    coefficient_spec: TransformSpec,
    selection: TransformSpec | None,
    taper: float | None,
) -> NDArray[np.float64]:
    """Return a response over a stored coefficient domain.

    ``selection`` may be narrower than ``coefficient_spec``.  This is the
    coefficient-domain counterpart of the one-shot filtering path and lets an
    analyzed field apply several selections without another analysis.
    """
    selected = coefficient_spec if selection is None else selection
    if not _spectral_selection_is_within(coefficient_spec, selected):
        raise ValueError("requested spectral selection exceeds the analyzed domain")
    degrees = alm_degrees(coefficient_spec.lmax, coefficient_spec.mmax)
    orders = alm_orders(coefficient_spec.lmax, coefficient_spec.mmax)
    weights = np.zeros(degrees.size, dtype=np.float64)
    inside = (degrees >= selected.lmin) & (degrees <= selected.lmax)
    inside &= orders <= selected.mmax
    if selected.truncation == "rhomboidal":
        inside &= degrees - orders <= selected.lmax - selected.mmax
    if taper is None:
        weights[inside] = 1.0
    elif selected.lmax == 0:
        # The published l(l+1) expression is singular at lmax=0.  Its only
        # retained endpoint is defined directly by the requested response.
        weights[inside] = taper
    else:
        coefficient = -np.log(taper) / (selected.lmax * (selected.lmax + 1)) ** 2
        degree_values = degrees[inside].astype(np.float64)
        weights[inside] = np.exp(
            -coefficient * (degree_values * (degree_values + 1.0)) ** 2
        )
    return weights


def _spectral_selection_is_within(
    analyzed: TransformSpec, selection: TransformSpec | None
) -> bool:
    """Whether every mode in ``selection`` is present in ``analyzed``."""
    if selection is None:
        return True
    if selection == analyzed:
        return True

    if selection.lmin < analyzed.lmin:
        return False
    if selection.lmax > analyzed.lmax:
        return False
    if selection.mmax > analyzed.mmax:
        return False

    if analyzed.truncation == "rhomboidal":
        # A non-rhomboidal selection contains (lmax, 0), whereas the
        # rhomboidal boundary contains (lmax, mmax).  These are the maximum
        # l-m envelopes of the two possible selection shapes.
        selection_l_minus_m_max = (
            selection.lmax - selection.mmax
            if selection.truncation == "rhomboidal"
            else selection.lmax
        )
        if selection_l_minus_m_max > analyzed.lmax - analyzed.mmax:
            return False
    return True


def _degree_scale(spec: TransformSpec, radius: float) -> NDArray[np.float64]:
    """Return the positive-degree vector-transform scale by packed mode."""
    degrees = alm_degrees(spec.lmax, spec.mmax).astype(np.float64)
    return np.sqrt(degrees * (degrees + 1.0)) / radius


def _laplacian_multiplier(spec: TransformSpec, radius: float) -> NDArray[np.float64]:
    """Return the scalar/vector Laplacian multiplier by packed mode."""
    degrees = alm_degrees(spec.lmax, spec.mmax).astype(np.float64)
    return -(degrees * (degrees + 1.0)) / radius**2


def _inverse_laplacian_multiplier(
    spec: TransformSpec, radius: float
) -> NDArray[np.float64]:
    """Return the zero-mode-defined inverse Laplacian multiplier."""
    degrees = alm_degrees(spec.lmax, spec.mmax).astype(np.float64)
    multiplier = np.zeros_like(degrees)
    positive = degrees > 0.0
    multiplier[positive] = -(radius**2) / (
        degrees[positive] * (degrees[positive] + 1.0)
    )
    return multiplier


def scalar_transform(
    field: xr.DataArray,
    source: FieldLayout,
    target: FieldLayout,
    transform: Callable[[NDArray[np.generic]], NDArray[np.float64]],
) -> xr.DataArray:
    """Apply a two-dimensional scalar kernel over all xarray leading dimensions."""
    canonical = source.canonicalize(field)
    output_changes_shape = (
        source.latitude_dim != target.latitude_dim
        or source.longitude_dim != target.longitude_dim
        or source.grid.nlat != target.grid.nlat
        or source.grid.nlon != target.grid.nlon
    )
    output_sizes = (
        {target.latitude_dim: target.grid.nlat, target.longitude_dim: target.grid.nlon}
        if output_changes_shape
        else None
    )

    def kernel(frame: NDArray[np.generic]) -> NDArray[np.float64]:
        return transform(frame)

    options = apply_ufunc_options(canonical, output_sizes=output_sizes)
    excluded_dimensions = (
        {source.latitude_dim, source.longitude_dim} if output_changes_shape else set()
    )
    result = xr.apply_ufunc(
        kernel,
        canonical,
        input_core_dims=[[source.latitude_dim, source.longitude_dim]],
        output_core_dims=[[target.latitude_dim, target.longitude_dim]],
        vectorize=True,
        output_dtypes=[np.float64],
        exclude_dims=excluded_dimensions,
        **options,
    )
    return restore_output(
        cast(xr.DataArray, result),
        source=source,
        target=target,
        original_dims=field.dims,
    )


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
        return parse_spectral(truncation)
    if lmin is None and lmax is None:
        return None
    if lmin is None or lmax is None:
        raise ValueError("explicit spectral bounds require both lmin= and lmax=")
    return TransformSpec(lmin, lmax, lmax)


def _validate_taper(taper: float | None) -> None:
    if taper is None:
        return
    if isinstance(taper, bool) or not isinstance(taper, (int, float)):
        raise TypeError("taper must be a finite float in (0, 1]")
    if not np.isfinite(taper) or taper <= 0.0 or taper > 1.0:
        raise ValueError("taper must be a finite value in (0, 1]")


def _validate_nontriangular_capabilities(
    selection: TransformSpec,
    source: GridCapabilities,
    target: GridCapabilities,
) -> None:
    """Validate independent latitude and longitude limits for shaped domains."""
    requested_mmax = selection.mmax
    for name, capabilities in (("source", source), ("target", target)):
        if selection.lmax > capabilities.latitude_lmax:
            raise ValueError(
                f"requested lmax={selection.lmax} exceeds the {name} latitude "
                f"bandwidth {capabilities.latitude_lmax}"
            )
        if requested_mmax > capabilities.longitude_mmax:
            raise ValueError(
                f"requested mmax={requested_mmax} exceeds the {name} longitude "
                f"bandwidth {capabilities.longitude_mmax}"
            )
