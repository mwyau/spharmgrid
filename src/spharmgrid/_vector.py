"""Private xarray adapters for paired tangent-vector SHT operations."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ._xarray import (
    FieldLayout,
    apply_ufunc_options,
    exact_align,
    field_layout,
    require_dataarray,
    resolve_nthreads,
    restore_output,
)
from .grids import grids_equivalent

VectorScalarKernel = Callable[
    [NDArray[np.generic], NDArray[np.generic], int], NDArray[np.float64]
]
VectorPairKernel = Callable[
    [NDArray[np.generic], NDArray[np.generic], int],
    tuple[NDArray[np.float64], NDArray[np.float64]],
]
VectorQuadKernel = Callable[
    [NDArray[np.generic], NDArray[np.generic], int],
    tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ],
]


def vector_inputs(
    u: xr.DataArray, v: xr.DataArray
) -> tuple[FieldLayout, xr.DataArray, xr.DataArray]:
    """Validate, canonicalize, and exactly align a geographic vector pair."""
    u = require_dataarray(u, "u")
    v = require_dataarray(v, "v")
    layout = field_layout(u)
    v_layout = field_layout(v)
    _require_matching_layouts(layout, v_layout, "u", "v")
    canonical_u = layout.canonicalize(u)
    canonical_v = v_layout.canonicalize(v)
    return layout, *exact_align(canonical_u, canonical_v, names=("u", "v"))


def vector_scalar_transform(
    original: xr.DataArray,
    source: FieldLayout,
    canonical_u: xr.DataArray,
    canonical_v: xr.DataArray,
    transform: VectorScalarKernel,
    *,
    nthreads: int | None,
) -> xr.DataArray:
    """Apply one paired-vector kernel that returns a scalar map."""
    dask_field = _dask_input(canonical_u, canonical_v)
    threads = resolve_nthreads(nthreads, dask_backed=dask_field.chunks is not None)

    def kernel(
        frame_u: NDArray[np.generic], frame_v: NDArray[np.generic]
    ) -> NDArray[np.float64]:
        return transform(frame_u, frame_v, threads)

    output = xr.apply_ufunc(
        kernel,
        canonical_u,
        canonical_v,
        input_core_dims=[
            [source.latitude_dim, source.longitude_dim],
            [source.latitude_dim, source.longitude_dim],
        ],
        output_core_dims=[[source.latitude_dim, source.longitude_dim]],
        vectorize=True,
        output_dtypes=[np.float64],
        **apply_ufunc_options(dask_field),
    )
    return restore_output(
        cast(xr.DataArray, output),
        source=source,
        target=source,
        original_dims=original.dims,
    )


def vector_pair_transform(
    original: xr.DataArray,
    source: FieldLayout,
    target: FieldLayout,
    canonical_u: xr.DataArray,
    canonical_v: xr.DataArray,
    transform: VectorPairKernel,
    *,
    nthreads: int | None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Apply one paired-vector kernel that returns two target-grid maps."""
    dask_field = _dask_input(canonical_u, canonical_v)
    threads = resolve_nthreads(nthreads, dask_backed=dask_field.chunks is not None)
    output_changes_shape = _output_changes_shape(source, target)
    output_sizes = (
        {target.latitude_dim: target.grid.nlat, target.longitude_dim: target.grid.nlon}
        if output_changes_shape
        else None
    )

    def kernel(
        frame_u: NDArray[np.generic], frame_v: NDArray[np.generic]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return transform(frame_u, frame_v, threads)

    outputs = xr.apply_ufunc(
        kernel,
        canonical_u,
        canonical_v,
        input_core_dims=[
            [source.latitude_dim, source.longitude_dim],
            [source.latitude_dim, source.longitude_dim],
        ],
        output_core_dims=[
            [target.latitude_dim, target.longitude_dim],
            [target.latitude_dim, target.longitude_dim],
        ],
        vectorize=True,
        output_dtypes=[np.float64, np.float64],
        exclude_dims=(
            {source.latitude_dim, source.longitude_dim}
            if output_changes_shape
            else set()
        ),
        **apply_ufunc_options(dask_field, output_sizes=output_sizes),
    )
    first, second = cast(tuple[xr.DataArray, xr.DataArray], outputs)
    return (
        restore_output(
            first,
            source=source,
            target=target,
            original_dims=original.dims,
        ),
        restore_output(
            second,
            source=source,
            target=target,
            original_dims=original.dims,
        ),
    )


def vector_quad_transform(
    original: xr.DataArray,
    source: FieldLayout,
    canonical_u: xr.DataArray,
    canonical_v: xr.DataArray,
    transform: VectorQuadKernel,
    *,
    nthreads: int | None,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Apply one paired-vector kernel that returns four source-grid maps."""
    dask_field = _dask_input(canonical_u, canonical_v)
    threads = resolve_nthreads(nthreads, dask_backed=dask_field.chunks is not None)

    def kernel(
        frame_u: NDArray[np.generic], frame_v: NDArray[np.generic]
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        return transform(frame_u, frame_v, threads)

    outputs = xr.apply_ufunc(
        kernel,
        canonical_u,
        canonical_v,
        input_core_dims=[
            [source.latitude_dim, source.longitude_dim],
            [source.latitude_dim, source.longitude_dim],
        ],
        output_core_dims=[
            [source.latitude_dim, source.longitude_dim],
            [source.latitude_dim, source.longitude_dim],
            [source.latitude_dim, source.longitude_dim],
            [source.latitude_dim, source.longitude_dim],
        ],
        vectorize=True,
        output_dtypes=[np.float64, np.float64, np.float64, np.float64],
        **apply_ufunc_options(dask_field),
    )
    first, second, third, fourth = cast(
        tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray], outputs
    )
    return (
        restore_output(
            first,
            source=source,
            target=source,
            original_dims=original.dims,
        ),
        restore_output(
            second,
            source=source,
            target=source,
            original_dims=original.dims,
        ),
        restore_output(
            third,
            source=source,
            target=source,
            original_dims=original.dims,
        ),
        restore_output(
            fourth,
            source=source,
            target=source,
            original_dims=original.dims,
        ),
    )


def _dask_input(first: xr.DataArray, second: xr.DataArray) -> xr.DataArray:
    return first if first.chunks is not None else second


def _output_changes_shape(source: FieldLayout, target: FieldLayout) -> bool:
    return (
        source.latitude_dim != target.latitude_dim
        or source.longitude_dim != target.longitude_dim
        or source.grid.nlat != target.grid.nlat
        or source.grid.nlon != target.grid.nlon
    )


def _require_matching_layouts(
    first: FieldLayout, second: FieldLayout, first_name: str, second_name: str
) -> None:
    if (
        first.latitude_dim != second.latitude_dim
        or first.longitude_dim != second.longitude_dim
        or first.coordinates.latitude_name != second.coordinates.latitude_name
        or first.coordinates.longitude_name != second.coordinates.longitude_name
    ):
        raise ValueError(
            f"{first_name} and {second_name} must use the same horizontal "
            "coordinate names and dimensions"
        )
    if not grids_equivalent(first.grid, second.grid):
        raise ValueError(
            f"{first_name} and {second_name} must describe equivalent supported grids"
        )
