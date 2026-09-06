"""Spectral scalar differential operators on supported global grids."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ._ducc import (
    alm_degrees,
    geometry_for,
    scalar_analysis,
    scalar_derivative_synthesis,
    scalar_synthesis,
    vector_analysis,
)
from ._vector import vector_inputs, vector_scalar_transform
from ._xarray import (
    FieldLayout,
    apply_ufunc_options,
    field_layout,
    require_dataarray,
    restore_output,
)
from .metadata import gradient_metadata, inverse_gradient_metadata, operator_metadata
from .spectral import scalar_transform, transform_spec

EARTH_RADIUS_M = 6_371_220.0


def gradient(
    field: xr.DataArray,
    *,
    eastward: str = "gradient_eastward",
    northward: str = "gradient_northward",
    radius: float = EARTH_RADIUS_M,
) -> xr.Dataset:
    """Return physical eastward and northward horizontal gradient components.

    The components have input units per metre when the input declares units.
    DUCC's spin-1 derivative synthesis supplies angular derivatives; this
    wrapper converts its southward-theta component to northward latitude.
    """
    field = require_dataarray(field)
    _validate_radius(radius)
    _validate_component_names(eastward, northward)
    source = field_layout(field)
    spec = transform_spec(source.grid, source.grid, None)

    def transform(
        frame: NDArray[np.generic],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        alm = scalar_analysis(
            frame,
            spec=spec,
            geometry=geometry_for(source.grid),
            phi0=source.transform_layout.phi0_radians,
        )
        derivatives = scalar_derivative_synthesis(
            alm,
            spec=spec,
            geometry=geometry_for(source.grid),
            ntheta=source.grid.nlat,
            nphi=source.grid.nlon,
            phi0=source.transform_layout.phi0_radians,
        )
        return derivatives[1] / radius, -derivatives[0] / radius

    east, north = _two_scalar_outputs(field, source, transform)
    east.name = eastward
    north.name = northward
    east.attrs = gradient_metadata(field, "eastward")
    north.attrs = gradient_metadata(field, "northward")
    return xr.Dataset({eastward: east, northward: north})


def inverse_gradient(
    eastward: xr.DataArray,
    northward: xr.DataArray,
    *,
    output: str | None = None,
    radius: float = EARTH_RADIUS_M,
) -> xr.DataArray:
    """Recover a scalar potential from a horizontal gradient vector field.

    The degree-zero scalar coefficient is set to zero because an additive
    constant cannot be recovered.  As in SPHEREPACK's inverse-gradient
    routines, rotational input is projected away: the returned potential has
    a gradient equal to the input field's irrotational component.
    """
    _validate_radius(radius)
    _validate_output_name(output)
    source, canonical_eastward, canonical_northward = vector_inputs(eastward, northward)
    spec = transform_spec(source.grid, source.grid, None)
    if spec.lmax < 1:
        raise ValueError("inverse gradient requires a grid supporting total degree l=1")
    degrees = alm_degrees(spec.lmax, spec.mmax).astype(np.float64)
    scale = np.sqrt(degrees * (degrees + 1.0)) / radius

    def transform(
        frame_eastward: NDArray[np.generic],
        frame_northward: NDArray[np.generic],
    ) -> NDArray[np.float64]:
        vector_alm = vector_analysis(
            frame_eastward,
            frame_northward,
            spec=spec,
            geometry=geometry_for(source.grid),
            phi0=source.transform_layout.phi0_radians,
        )
        potential_alm = np.zeros_like(vector_alm[0])
        nonzero = scale > 0.0
        potential_alm[nonzero] = vector_alm[0, nonzero] / scale[nonzero]
        return scalar_synthesis(
            potential_alm[np.newaxis, :],
            spec=spec,
            geometry=geometry_for(source.grid),
            ntheta=source.grid.nlat,
            nphi=source.grid.nlon,
            phi0=source.transform_layout.phi0_radians,
        )

    result = vector_scalar_transform(
        eastward,
        source,
        canonical_eastward,
        canonical_northward,
        transform,
    )
    result.name = output
    result.attrs = inverse_gradient_metadata(eastward, northward)
    return result


def laplacian(
    field: xr.DataArray,
    *,
    radius: float = EARTH_RADIUS_M,
) -> xr.DataArray:
    """Apply the physical spherical Laplacian to a scalar field.

    Each total degree is multiplied by ``-l(l+1)/radius**2``.
    """
    field = require_dataarray(field)
    _validate_radius(radius)
    source = field_layout(field)
    spec = transform_spec(source.grid, source.grid, None)
    degrees = alm_degrees(spec.lmax, spec.mmax).astype(np.float64)
    multiplier = -(degrees * (degrees + 1.0)) / radius**2

    def transform(frame: NDArray[np.generic]) -> NDArray[np.float64]:
        alm = scalar_analysis(
            frame,
            spec=spec,
            geometry=geometry_for(source.grid),
            phi0=source.transform_layout.phi0_radians,
        )
        result = alm * multiplier[np.newaxis, :]
        return scalar_synthesis(
            result,
            spec=spec,
            geometry=geometry_for(source.grid),
            ntheta=source.grid.nlat,
            nphi=source.grid.nlon,
            phi0=source.transform_layout.phi0_radians,
        )

    result = scalar_transform(field, source, source, transform)
    result.name = field.name
    result.attrs = operator_metadata(field, "laplacian")
    return result


def inverse_laplacian(
    field: xr.DataArray,
    *,
    radius: float = EARTH_RADIUS_M,
) -> xr.DataArray:
    """Solve the spherical inverse Laplacian with its degree-zero mode set to zero.

    The additive constant is undefined.  spharmgrid returns the zero-mean
    spectral solution by applying ``-radius**2/(l(l+1))`` only for ``l > 0``.
    """
    field = require_dataarray(field)
    _validate_radius(radius)
    source = field_layout(field)
    spec = transform_spec(source.grid, source.grid, None)
    degrees = alm_degrees(spec.lmax, spec.mmax).astype(np.float64)
    multiplier = np.zeros_like(degrees)
    nonzero = degrees > 0.0
    multiplier[nonzero] = -(radius**2) / (degrees[nonzero] * (degrees[nonzero] + 1.0))

    def transform(frame: NDArray[np.generic]) -> NDArray[np.float64]:
        alm = scalar_analysis(
            frame,
            spec=spec,
            geometry=geometry_for(source.grid),
            phi0=source.transform_layout.phi0_radians,
        )
        result = alm * multiplier[np.newaxis, :]
        return scalar_synthesis(
            result,
            spec=spec,
            geometry=geometry_for(source.grid),
            ntheta=source.grid.nlat,
            nphi=source.grid.nlon,
            phi0=source.transform_layout.phi0_radians,
        )

    result = scalar_transform(field, source, source, transform)
    result.name = field.name
    result.attrs = operator_metadata(field, "inverse_laplacian")
    return result


def _two_scalar_outputs(
    field: xr.DataArray,
    source: FieldLayout,
    transform: Callable[
        [NDArray[np.generic]], tuple[NDArray[np.float64], NDArray[np.float64]]
    ],
) -> tuple[xr.DataArray, xr.DataArray]:
    """Apply one analysis-sharing scalar kernel with two map outputs."""
    canonical = source.canonicalize(field)

    def kernel(
        frame: NDArray[np.generic],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return transform(frame)

    result = xr.apply_ufunc(
        kernel,
        canonical,
        input_core_dims=[[source.latitude_dim, source.longitude_dim]],
        output_core_dims=[
            [source.latitude_dim, source.longitude_dim],
            [source.latitude_dim, source.longitude_dim],
        ],
        vectorize=True,
        output_dtypes=[np.float64, np.float64],
        **apply_ufunc_options(canonical),
    )
    east, north = cast(tuple[xr.DataArray, xr.DataArray], result)
    return (
        restore_output(
            east,
            source=source,
            target=source,
            original_dims=field.dims,
        ),
        restore_output(
            north,
            source=source,
            target=source,
            original_dims=field.dims,
        ),
    )


def _validate_radius(radius: float) -> None:
    if isinstance(radius, bool) or not isinstance(radius, (int, float)):
        raise TypeError("radius must be a positive finite number in metres")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be a positive finite number in metres")


def _validate_component_names(eastward: str, northward: str) -> None:
    """Prevent a two-component gradient from silently overwriting one output."""
    if not isinstance(eastward, str) or not isinstance(northward, str):
        raise TypeError("gradient output names must be strings")
    if not eastward or not northward:
        raise ValueError("gradient output names must be non-empty")
    if eastward == northward:
        raise ValueError("gradient output names must be distinct")


def _validate_output_name(output: str | None) -> None:
    if output is None:
        return
    if not isinstance(output, str):
        raise TypeError("output must be a string or None")
    if not output:
        raise ValueError("output must be non-empty when provided")
