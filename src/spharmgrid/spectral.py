"""Scalar spherical-harmonic filtering primitives built on DUCC."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ._ducc import (
    TransformSpec,
    alm_degrees,
    geometry_for,
    scalar_analysis,
    scalar_synthesis,
)
from ._xarray import (
    FieldLayout,
    apply_ufunc_options,
    field_layout,
    require_dataarray,
    restore_output,
)
from .grids import Grid, grid_capabilities
from .metadata import preserve_quantity_metadata

_SPECTRAL_RE = re.compile(r"^t(?:(\d+)-)?(\d+)$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class SpectralRange:
    """Inclusive total-wavenumber bounds for a triangular spectral selection."""

    lmin: int
    lmax: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.lmin, bool)
            or isinstance(self.lmax, bool)
            or not isinstance(self.lmin, int)
            or not isinstance(self.lmax, int)
        ):
            raise TypeError("spectral bounds must be integers")
        if self.lmin < 0 or self.lmin > self.lmax:
            raise ValueError("spectral bounds must satisfy 0 <= lmin <= lmax")


def parse_spectral(value: str) -> SpectralRange:
    """Parse case-insensitive ``Tn`` or ``Tn-m`` notation.

    An en dash is accepted in place of the ASCII hyphen.  The notation is a
    total-wavenumber range, not a grid name.
    """
    if not isinstance(value, str):
        raise TypeError("spectral notation must be a string")
    normalized = value.strip().replace("–", "-")
    match = _SPECTRAL_RE.fullmatch(normalized)
    if match is None:
        raise ValueError("spectral notation must be Tn or Tn-m, for example T42")
    lower, upper = match.groups()
    return SpectralRange(0 if lower is None else int(lower), int(upper))


def filter(
    field: xr.DataArray,
    truncation: str | SpectralRange | None = None,
    *,
    lmin: int | None = None,
    lmax: int | None = None,
    taper: float | None = None,
) -> xr.DataArray:
    """Apply a hard or Sardeshmukh–Hoskins tapered spectral selection.

    With ``taper=None`` (the default), all modes inside the selected inclusive
    range are unchanged and all other modes are zero.  ``taper`` is the
    response at the upper retained total wavenumber.
    """
    field = require_dataarray(field)
    source = field_layout(field)
    selection = _resolve_spectral_range(truncation, lmin=lmin, lmax=lmax)
    spec = transform_spec(source.grid, source.grid, selection)
    retained = selection or SpectralRange(0, spec.lmax)
    _validate_taper(taper)

    def transform(frame: NDArray[np.generic]) -> NDArray[np.float64]:
        alm = scalar_analysis(
            frame,
            spec=spec,
            geometry=geometry_for(source.grid),
            phi0=source.transform_layout.phi0_radians,
        )
        filtered = apply_spectral_selection(alm, spec, retained, taper)
        return scalar_synthesis(
            filtered,
            spec=spec,
            geometry=geometry_for(source.grid),
            ntheta=source.grid.nlat,
            nphi=source.grid.nlon,
            phi0=source.transform_layout.phi0_radians,
        )

    result = scalar_transform(field, source, source, transform)
    return preserve_quantity_metadata(result, field)


def transform_spec(
    source: Grid,
    target: Grid,
    selection: SpectralRange | None,
) -> TransformSpec:
    """Choose a DUCC bandwidth without silently clamping explicit ``Tn`` input."""
    source_capabilities = grid_capabilities(source)
    target_capabilities = grid_capabilities(target)
    if selection is not None:
        supported = min(
            source_capabilities.triangular_lmax,
            target_capabilities.triangular_lmax,
        )
        if selection.lmax > supported:
            raise ValueError(
                f"requested lmax={selection.lmax} exceeds the supported triangular "
                f"bandwidth {supported} for the source and target grids"
            )
        return TransformSpec(selection.lmax, selection.lmax)

    # Without a T-range the source and target can retain high-degree zonally
    # symmetric content when latitude sampling permits it, while m remains
    # bounded by both longitude samplings.
    lmax = min(source_capabilities.latitude_lmax, target_capabilities.latitude_lmax)
    mmax = min(
        lmax,
        source_capabilities.longitude_mmax,
        target_capabilities.longitude_mmax,
    )
    return TransformSpec(lmax, mmax)


def apply_spectral_selection(
    alm: NDArray[np.complexfloating],
    spec: TransformSpec,
    selection: SpectralRange,
    taper: float | None,
) -> NDArray[np.complex128]:
    """Return scalar coefficients masked to a retained degree range."""
    if selection.lmax > spec.lmax:
        raise ValueError("spectral selection exceeds transform lmax")
    degrees = alm_degrees(spec.lmax, spec.mmax)
    weights = np.zeros(degrees.size, dtype=np.float64)
    inside = (degrees >= selection.lmin) & (degrees <= selection.lmax)
    if taper is None:
        weights[inside] = 1.0
    elif selection.lmax == 0:
        # The published l(l+1) expression is singular at lmax=0.  Its only
        # retained endpoint is defined directly by the requested response.
        weights[inside] = taper
    else:
        coefficient = -np.log(taper) / (selection.lmax * (selection.lmax + 1)) ** 2
        degree_values = degrees[inside].astype(np.float64)
        weights[inside] = np.exp(
            -coefficient * (degree_values * (degree_values + 1.0)) ** 2
        )
    result = np.array(alm, dtype=np.complex128, copy=True)
    result *= weights[np.newaxis, :]
    return result


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


def _resolve_spectral_range(
    truncation: str | SpectralRange | None,
    *,
    lmin: int | None,
    lmax: int | None,
) -> SpectralRange | None:
    if truncation is not None and (lmin is not None or lmax is not None):
        raise ValueError("use either truncation= or explicit lmin= and lmax=, not both")
    if truncation is not None:
        if isinstance(truncation, SpectralRange):
            return truncation
        return parse_spectral(truncation)
    if lmin is None and lmax is None:
        return None
    if lmin is None or lmax is None:
        raise ValueError("explicit spectral bounds require both lmin= and lmax=")
    return SpectralRange(lmin, lmax)


def _validate_taper(taper: float | None) -> None:
    if taper is None:
        return
    if isinstance(taper, bool) or not isinstance(taper, (int, float)):
        raise TypeError("taper must be a finite float in (0, 1]")
    if not np.isfinite(taper) or taper <= 0.0 or taper > 1.0:
        raise ValueError("taper must be a finite value in (0, 1]")
