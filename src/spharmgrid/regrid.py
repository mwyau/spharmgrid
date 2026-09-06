"""Spectral regridding among supported global GL and CC grids."""

from __future__ import annotations

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ._ducc import (
    geometry_for,
    scalar_analysis,
    scalar_synthesis,
    vector_analysis,
    vector_synthesis,
)
from ._vector import vector_inputs, vector_pair_transform
from ._xarray import field_layout, require_dataarray, target_layout
from .grids import Grid
from .metadata import preserve_quantity_metadata
from .spectral import (
    SpectralRange,
    _resolve_spectral_range,
    _validate_taper,
    apply_spectral_selection,
    scalar_transform,
    transform_spec,
)


def regrid(
    field: xr.DataArray,
    target_grid: Grid | xr.DataArray | xr.Dataset,
    spectral: str | SpectralRange | None = None,
    *,
    lmin: int | None = None,
    lmax: int | None = None,
    taper: float | None = None,
    nthreads: int | None = None,
) -> xr.DataArray:
    """Spectrally regrid a GL or CC field, optionally filtering in one cycle.

    The operation analyzes the source once, applies the optional retained
    range/taper to those coefficients, and synthesizes once on the target.
    ``Tn`` is a truncation, not a target-grid shorthand.
    """
    field = require_dataarray(field)
    source = field_layout(field)
    target = target_layout(target_grid, source)
    selection = _resolve_spectral_range(spectral, lmin=lmin, lmax=lmax)
    spec = transform_spec(source.grid, target.grid, selection)
    retained = selection or SpectralRange(0, spec.lmax)
    _validate_taper(taper)

    def transform(frame: NDArray[np.generic], threads: int) -> NDArray[np.float64]:
        alm = scalar_analysis(
            frame,
            spec=spec,
            geometry=geometry_for(source.grid),
            phi0=source.transform_layout.phi0_radians,
            nthreads=threads,
        )
        if selection is not None or taper is not None:
            alm = apply_spectral_selection(alm, spec, retained, taper)
        return scalar_synthesis(
            alm,
            spec=spec,
            geometry=geometry_for(target.grid),
            ntheta=target.grid.nlat,
            nphi=target.grid.nlon,
            phi0=target.transform_layout.phi0_radians,
            nthreads=threads,
        )

    result = scalar_transform(field, source, target, transform, nthreads=nthreads)
    return preserve_quantity_metadata(result, field)


def regrid_vector(
    u: xr.DataArray,
    v: xr.DataArray,
    target_grid: Grid | xr.DataArray | xr.Dataset,
    spectral: str | SpectralRange | None = None,
    *,
    lmin: int | None = None,
    lmax: int | None = None,
    taper: float | None = None,
    eastward: str = "u",
    northward: str = "v",
    nthreads: int | None = None,
) -> xr.Dataset:
    """Spectrally regrid geographic vector components in one spin-1 cycle.

    The operation analyzes eastward/northward components together, applies an
    optional degree selection or taper to both vector-harmonic coefficient
    families, then synthesizes both components on the target grid.  It does
    not scalar-regrid the two physical components independently.
    """
    _validate_component_names(eastward, northward)
    source, canonical_u, canonical_v = vector_inputs(u, v)
    target = target_layout(target_grid, source)
    selection = _resolve_spectral_range(spectral, lmin=lmin, lmax=lmax)
    spec = transform_spec(source.grid, target.grid, selection)
    if spec.lmax < 1:
        raise ValueError(
            "vector regridding requires a grid supporting total degree l=1"
        )
    retained = selection or SpectralRange(0, spec.lmax)
    _validate_taper(taper)

    def transform(
        frame_u: NDArray[np.generic], frame_v: NDArray[np.generic], threads: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        alm = vector_analysis(
            frame_u,
            frame_v,
            spec=spec,
            geometry=geometry_for(source.grid),
            phi0=source.transform_layout.phi0_radians,
            nthreads=threads,
        )
        if selection is not None or taper is not None:
            alm = apply_spectral_selection(alm, spec, retained, taper)
        return vector_synthesis(
            alm[0],
            alm[1],
            spec=spec,
            geometry=geometry_for(target.grid),
            ntheta=target.grid.nlat,
            nphi=target.grid.nlon,
            phi0=target.transform_layout.phi0_radians,
            nthreads=threads,
        )

    output_u, output_v = vector_pair_transform(
        u,
        source,
        target,
        canonical_u,
        canonical_v,
        transform,
        nthreads=nthreads,
    )
    output_u = _preserve_component_metadata(output_u, u, eastward)
    output_v = _preserve_component_metadata(output_v, v, northward)
    return xr.Dataset({eastward: output_u, northward: output_v})


def _preserve_component_metadata(
    result: xr.DataArray, source: xr.DataArray, name: str
) -> xr.DataArray:
    output = result.copy(deep=False)
    output.name = name
    output.attrs = dict(source.attrs)
    return output


def _validate_component_names(eastward: str, northward: str) -> None:
    if not isinstance(eastward, str) or not isinstance(northward, str):
        raise TypeError("vector output names must be strings")
    if not eastward or not northward:
        raise ValueError("vector output names must be non-empty")
    if eastward == northward:
        raise ValueError("vector output names must be distinct")
