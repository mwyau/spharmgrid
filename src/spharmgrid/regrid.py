"""Spectral regridding among supported global GL and CC grids."""

from __future__ import annotations

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ._ducc import geometry_for, scalar_analysis, scalar_synthesis
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
