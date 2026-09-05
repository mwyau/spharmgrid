"""Internal xarray adapters shared by scalar and vector operations."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any, Literal, NotRequired, TypedDict, TypeVar

import xarray as xr

from .grids import (
    Grid,
    GridLayout,
    HorizontalCoordinates,
    coordinate_attributes,
    find_horizontal_coordinates,
    grid_layout,
)

_ArrayFunction = TypeVar("_ArrayFunction", bound=Callable[..., object])


class ApplyUfuncDaskOptions(TypedDict):
    """The narrow Dask option subset passed through to xarray.apply_ufunc."""

    dask: Literal["forbidden", "parallelized"]
    dask_gufunc_kwargs: NotRequired[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class FieldLayout:
    """An xarray field's grid, coordinate names, and DUCC canonical order."""

    grid: Grid
    coordinates: HorizontalCoordinates
    transform_layout: GridLayout
    latitude_coordinate: xr.DataArray
    longitude_coordinate: xr.DataArray

    @property
    def latitude_dim(self) -> str:
        return self.coordinates.latitude_dim

    @property
    def longitude_dim(self) -> str:
        return self.coordinates.longitude_dim

    def canonicalize(self, field: xr.DataArray) -> xr.DataArray:
        """Move latitude and longitude data together into DUCC order."""
        result = field.isel(
            {
                self.latitude_dim: self.transform_layout.latitude.canonical_indices,
                self.longitude_dim: self.transform_layout.longitude.canonical_indices,
            }
        )
        # Canonical labels make equivalent [-180, 180) and [0, 360) inputs
        # align before a multi-field transform.  User-facing coordinates are
        # restored by ``restore`` after synthesis.
        return result.assign_coords(
            {
                self.coordinates.latitude_name: (
                    self.latitude_dim,
                    self.transform_layout.latitude.canonical_values,
                ),
                self.coordinates.longitude_name: (
                    self.longitude_dim,
                    self.transform_layout.longitude.canonical_values,
                ),
            }
        )

    def restore(self, field: xr.DataArray) -> xr.DataArray:
        """Restore this layout's original coordinate order and values."""
        result = field.isel(
            {
                self.latitude_dim: self.transform_layout.latitude.restore_indices,
                self.longitude_dim: self.transform_layout.longitude.restore_indices,
            }
        )
        return result.assign_coords(
            {
                self.coordinates.latitude_name: self.latitude_coordinate.copy(
                    deep=False
                ),
                self.coordinates.longitude_name: self.longitude_coordinate.copy(
                    deep=False
                ),
            }
        )


def field_layout(field: xr.DataArray) -> FieldLayout:
    """Inspect a field and construct its supported grid/layout description."""
    from .grids import detect_grid

    coordinates = find_horizontal_coordinates(field)
    grid = detect_grid(field)
    return FieldLayout(
        grid=grid,
        coordinates=coordinates,
        transform_layout=grid_layout(grid),
        latitude_coordinate=field.coords[coordinates.latitude_name],
        longitude_coordinate=field.coords[coordinates.longitude_name],
    )


def target_layout(
    target: Grid | xr.DataArray | xr.Dataset,
    source: FieldLayout,
) -> FieldLayout:
    """Build the output layout for a grid descriptor or xarray reference."""
    if isinstance(target, Grid):
        latitude_dim = source.latitude_dim
        longitude_dim = source.longitude_dim
        latitude_name = source.coordinates.latitude_name
        longitude_name = source.coordinates.longitude_name
        latitude = xr.DataArray(
            target.latitude,
            dims=(latitude_dim,),
            name=latitude_name,
            attrs=coordinate_attributes("latitude"),
        )
        longitude = xr.DataArray(
            target.longitude,
            dims=(longitude_dim,),
            name=longitude_name,
            attrs=coordinate_attributes("longitude"),
        )
        coordinates = HorizontalCoordinates(
            latitude_name=latitude_name,
            longitude_name=longitude_name,
            latitude_dim=latitude_dim,
            longitude_dim=longitude_dim,
        )
        return FieldLayout(
            grid=target,
            coordinates=coordinates,
            transform_layout=grid_layout(target),
            latitude_coordinate=latitude,
            longitude_coordinate=longitude,
        )

    from .grids import detect_grid

    coordinates = find_horizontal_coordinates(target)
    grid = detect_grid(target)
    return FieldLayout(
        grid=grid,
        coordinates=coordinates,
        transform_layout=grid_layout(grid),
        latitude_coordinate=target.coords[coordinates.latitude_name],
        longitude_coordinate=target.coords[coordinates.longitude_name],
    )


def apply_ufunc_options(
    field: xr.DataArray,
    *,
    output_sizes: dict[str, int] | None = None,
) -> ApplyUfuncDaskOptions:
    """Return Dask options that keep horizontal transforms lazy when possible."""
    if field.chunks is None:
        return {"dask": "forbidden"}
    # xarray's public ``dask_gufunc_kwargs`` annotation intentionally accepts
    # heterogeneous values. Keep that third-party boundary localized here.
    gufunc_kwargs: dict[str, Any] = {"allow_rechunk": True}
    if output_sizes is not None:
        gufunc_kwargs["output_sizes"] = output_sizes
    return {"dask": "parallelized", "dask_gufunc_kwargs": gufunc_kwargs}


def restore_output(
    output: xr.DataArray,
    *,
    source: FieldLayout,
    target: FieldLayout,
    original_dims: tuple[Hashable, ...],
) -> xr.DataArray:
    """Restore target coordinates and put horizontal dimensions in field order."""
    output = target.restore(output)
    desired_dims = tuple(
        target.latitude_dim
        if dimension == source.latitude_dim
        else target.longitude_dim
        if dimension == source.longitude_dim
        else dimension
        for dimension in original_dims
    )
    if len(set(desired_dims)) != len(desired_dims):
        raise ValueError(
            "target horizontal dimensions conflict with a non-spatial input dimension"
        )
    return output.transpose(*desired_dims)


def resolve_nthreads(nthreads: int | None, *, dask_backed: bool) -> int:
    """Resolve DUCC threads, avoiding nested default parallelism under Dask."""
    if nthreads is None:
        # DUCC's Python SHT functions currently default to one thread.  Keep
        # that documented default for eager calls and use the same safe choice
        # for independent Dask tasks.
        return 1
    if isinstance(nthreads, bool) or not isinstance(nthreads, int) or nthreads < 0:
        raise ValueError("nthreads must be a non-negative integer or None")
    if dask_backed and nthreads == 0:
        # Explicit zero is DUCC's documented request for all available threads;
        # honor it because the caller has made that oversubscription choice.
        return 0
    return nthreads


def require_dataarray(field: xr.DataArray, name: str = "field") -> xr.DataArray:
    """Reject bare arrays and datasets at the numerical API boundary."""
    if not isinstance(field, xr.DataArray):
        raise TypeError(f"{name} must be an xarray.DataArray")
    return field


def exact_align(
    first: xr.DataArray, second: xr.DataArray, *, names: tuple[str, str]
) -> tuple[xr.DataArray, xr.DataArray]:
    """Require identical named dimensions and non-spatial coordinate alignment."""
    if set(first.dims) != set(second.dims):
        raise ValueError(
            f"{names[0]} and {names[1]} must have the same dimensions; got "
            f"{first.dims!r} and {second.dims!r}"
        )
    second = second.transpose(*first.dims)
    try:
        aligned = xr.align(first, second, join="exact", copy=False)
    except ValueError as error:
        raise ValueError(
            f"{names[0]} and {names[1]} must have exactly aligned coordinates"
        ) from error
    return aligned
