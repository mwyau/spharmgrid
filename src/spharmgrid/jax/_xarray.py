# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Xarray/JAX preparation, execution, and output adapters."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from typing import TypeVar, cast, overload

import jax
import jax.numpy as jnp
import xarray as xr
from jax import Array, core

from .._xarray import FieldLayout, exact_align, field_layout, require_dataarray
from .._xarray import target_layout as _target_layout
from ..grids import Grid, grids_equivalent

_Result = TypeVar("_Result")
_Target = Grid | xr.DataArray | xr.Dataset


@dataclass(frozen=True, slots=True)
class _PreparedField:
    """A validated Xarray field and its trailing JAX data."""

    field: xr.DataArray
    layout: FieldLayout
    data: Array


@dataclass(frozen=True, slots=True)
class _PreparedPair:
    """Two aligned fields using the first field's restored layout."""

    first: xr.DataArray
    second: xr.DataArray
    layout: FieldLayout
    first_data: Array
    second_data: Array


def _require_jax_data(field: xr.DataArray, name: str) -> None:
    if not isinstance(field.data, (Array, core.Tracer)):
        raise TypeError(
            ".sgj requires JAX-backed data; use spharmgrid.jax.device_put(...) first"
        )


def _trailing_data(field: xr.DataArray, layout: FieldLayout) -> Array:
    leading_dims = tuple(
        dimension
        for dimension in field.dims
        if dimension not in (layout.latitude_dim, layout.longitude_dim)
    )
    return cast(
        Array,
        field.transpose(
            *leading_dims,
            layout.latitude_dim,
            layout.longitude_dim,
        ).data,
    )


def _prepare_field(field: xr.DataArray, name: str = "field") -> _PreparedField:
    field = require_dataarray(field, name)
    _require_jax_data(field, name)
    layout = field_layout(field)
    return _PreparedField(field, layout, _trailing_data(field, layout))


def _require_matching_layouts(
    first: FieldLayout,
    second: FieldLayout,
    names: tuple[str, str],
) -> None:
    if (
        first.latitude_dim != second.latitude_dim
        or first.longitude_dim != second.longitude_dim
        or first.coordinates.latitude_name != second.coordinates.latitude_name
        or first.coordinates.longitude_name != second.coordinates.longitude_name
    ):
        raise ValueError(
            f"{names[0]} and {names[1]} must use the same horizontal "
            "coordinate names and dimensions"
        )
    if not grids_equivalent(first.grid, second.grid):
        raise ValueError(
            f"{names[0]} and {names[1]} must describe equivalent supported grids"
        )


def _prepare_pair(
    first: xr.DataArray,
    second: xr.DataArray,
    *,
    names: tuple[str, str],
) -> _PreparedPair:
    first = require_dataarray(first, names[0])
    second = require_dataarray(second, names[1])
    _require_jax_data(first, names[0])
    _require_jax_data(second, names[1])
    first_layout = field_layout(first)
    second_layout = field_layout(second)
    _require_matching_layouts(first_layout, second_layout, names)

    canonical_first = first_layout.canonicalize(first)
    canonical_second = second_layout.canonicalize(second)
    canonical_first, canonical_second = exact_align(
        canonical_first,
        canonical_second,
        names=names,
    )
    # The backend receives both arrays in the first field's cyclic coordinate
    # order. The same layout must also be used when their outputs are wrapped.
    aligned_first = first_layout.restore(canonical_first)
    aligned_second = first_layout.restore(canonical_second)
    return _PreparedPair(
        first,
        second,
        first_layout,
        _trailing_data(aligned_first, first_layout),
        _trailing_data(aligned_second, first_layout),
    )


def _destination_layout(source: FieldLayout, target: _Target | None) -> FieldLayout:
    return source if target is None else _target_layout(target, source)


def _apply_scalar_operation(
    field: xr.DataArray,
    operation: Callable[..., _Result],
    *operation_args: object,
    target: _Target | None = None,
    **operation_kwargs: object,
) -> tuple[_PreparedField, FieldLayout, _Result]:
    """Prepare one field, call a raw scalar operation, and return its layout."""
    prepared = _prepare_field(field)
    destination = _destination_layout(prepared.layout, target)
    if target is None:
        result = operation(
            prepared.data,
            *operation_args,
            grid=prepared.layout.grid,
            **operation_kwargs,
        )
    else:
        result = operation(
            prepared.data,
            destination.grid,
            *operation_args,
            source_grid=prepared.layout.grid,
            **operation_kwargs,
        )
    return prepared, destination, result


def _apply_pair_operation(
    first: xr.DataArray,
    second: xr.DataArray,
    operation: Callable[..., _Result],
    *operation_args: object,
    names: tuple[str, str],
    target: _Target | None = None,
    **operation_kwargs: object,
) -> tuple[_PreparedPair, FieldLayout, _Result]:
    """Align a pair, call a raw vector operation, and return its layout."""
    prepared = _prepare_pair(first, second, names=names)
    destination = _destination_layout(prepared.layout, target)
    if target is None:
        result = operation(
            prepared.first_data,
            prepared.second_data,
            *operation_args,
            grid=prepared.layout.grid,
            **operation_kwargs,
        )
    else:
        result = operation(
            prepared.first_data,
            prepared.second_data,
            destination.grid,
            *operation_args,
            source_grid=prepared.layout.grid,
            **operation_kwargs,
        )
    return prepared, destination, result


def _apply_scalar(
    field: xr.DataArray,
    operation: Callable[..., Array],
    *operation_args: object,
    name: Hashable | None,
    attrs: Mapping[str, object] | None = None,
    target: _Target | None = None,
    **operation_kwargs: object,
) -> xr.DataArray:
    """Apply a scalar operation and wrap one labeled output."""
    prepared, destination, result = _apply_scalar_operation(
        field,
        operation,
        *operation_args,
        target=target,
        **operation_kwargs,
    )
    return _wrap(
        result,
        prepared.field,
        prepared.layout,
        destination,
        name=name,
        attrs=attrs,
    )


def _output_dims(
    template: xr.DataArray,
    source: FieldLayout,
    target: FieldLayout,
) -> tuple[Hashable, ...]:
    dimensions = tuple(
        target.latitude_dim
        if dimension == source.latitude_dim
        else target.longitude_dim
        if dimension == source.longitude_dim
        else dimension
        for dimension in template.dims
    )
    if len(set(dimensions)) != len(dimensions):
        raise ValueError(
            "target horizontal dimensions conflict with a non-spatial input dimension"
        )
    return dimensions


def _output_coordinates(
    template: xr.DataArray,
    source: FieldLayout,
    target: FieldLayout,
    dimensions: tuple[Hashable, ...],
    sizes: Mapping[Hashable, int],
) -> dict[Hashable, xr.DataArray]:
    horizontal_names = {
        source.coordinates.latitude_name,
        source.coordinates.longitude_name,
        target.coordinates.latitude_name,
        target.coordinates.longitude_name,
    }
    coordinates: dict[Hashable, xr.DataArray] = {}
    for name, coordinate in template.coords.items():
        if name in horizontal_names:
            continue
        mapped_dimensions = tuple(
            target.latitude_dim
            if dimension == source.latitude_dim
            else target.longitude_dim
            if dimension == source.longitude_dim
            else dimension
            for dimension in coordinate.dims
        )
        if not all(dimension in dimensions for dimension in mapped_dimensions):
            continue
        if any(
            coordinate.sizes[dimension] != sizes[mapped_dimension]
            for dimension, mapped_dimension in zip(
                coordinate.dims, mapped_dimensions, strict=True
            )
        ):
            continue
        rename = {
            dimension: mapped_dimension
            for dimension, mapped_dimension in zip(
                coordinate.dims, mapped_dimensions, strict=True
            )
            if dimension != mapped_dimension
        }
        coordinates[name] = coordinate.copy(deep=False).rename(rename)
    coordinates[target.coordinates.latitude_name] = target.latitude_coordinate.copy(
        deep=False
    )
    coordinates[target.coordinates.longitude_name] = target.longitude_coordinate.copy(
        deep=False
    )
    return coordinates


def _wrap(
    data: Array,
    template: xr.DataArray,
    source: FieldLayout,
    target: FieldLayout,
    *,
    name: Hashable | None,
    attrs: Mapping[str, object] | None = None,
) -> xr.DataArray:
    dimensions = _output_dims(template, source, target)
    canonical_dimensions = tuple(
        dimension
        for dimension in template.dims
        if dimension not in (source.latitude_dim, source.longitude_dim)
    ) + (target.latitude_dim, target.longitude_dim)
    axes = tuple(canonical_dimensions.index(dimension) for dimension in dimensions)
    if axes != tuple(range(len(axes))):
        data = jnp.transpose(data, axes)
    sizes = dict(zip(dimensions, data.shape, strict=True))
    return xr.DataArray(
        data,
        dims=dimensions,
        coords=_output_coordinates(template, source, target, dimensions, sizes),
        name=name,
        attrs=dict(template.attrs) if attrs is None else dict(attrs),
    )


def _dataset_outputs(
    outputs: tuple[Array, ...],
    template: xr.DataArray,
    source: FieldLayout,
    target: FieldLayout,
    *,
    names: tuple[str, ...],
    attrs: tuple[Mapping[str, object] | None, ...],
) -> xr.Dataset:
    """Wrap several numerical outputs using one structural field layout."""
    fields = {
        name: _wrap(
            data,
            template,
            source,
            target,
            name=name,
            attrs=output_attrs,
        )
        for data, name, output_attrs in zip(outputs, names, attrs, strict=True)
    }
    return xr.Dataset(fields)


@overload
def device_put(
    obj: xr.DataArray,
    device: jax.Device | jax.sharding.Sharding | None = None,
) -> xr.DataArray: ...


@overload
def device_put(
    obj: xr.Dataset,
    device: jax.Device | jax.sharding.Sharding | None = None,
) -> xr.Dataset: ...


def device_put(
    obj: xr.DataArray | xr.Dataset,
    device: jax.Device | jax.sharding.Sharding | None = None,
) -> xr.DataArray | xr.Dataset:
    """Put Xarray numerical payloads on a JAX device.

    Coordinates remain ordinary Xarray objects. Only a DataArray's data or a
    Dataset's data variables are passed to :func:`jax.device_put`.
    """
    if isinstance(obj, xr.DataArray):
        return _with_data(obj, _put_data(obj.data, "DataArray", obj.name, device))
    if isinstance(obj, xr.Dataset):
        result = obj.copy(deep=False)
        for name, variable in obj.data_vars.items():
            result[name].data = _put_data(
                variable.data,
                "Dataset data variable",
                name,
                device,
            )
        return result
    raise TypeError("device_put expects an xarray.DataArray or xarray.Dataset")


@overload
def device_get(obj: xr.DataArray) -> xr.DataArray: ...


@overload
def device_get(obj: xr.Dataset) -> xr.Dataset: ...


def device_get(obj: xr.DataArray | xr.Dataset) -> xr.DataArray | xr.Dataset:
    """Copy Xarray numerical payloads from a JAX device to host memory."""
    if isinstance(obj, xr.DataArray):
        return _with_data(obj, jax.device_get(obj.data))
    if isinstance(obj, xr.Dataset):
        result = obj.copy(deep=False)
        for name, variable in obj.data_vars.items():
            result[name].data = jax.device_get(variable.data)
        return result
    raise TypeError("device_get expects an xarray.DataArray or xarray.Dataset")


def _put_data(
    data: object,
    container: str,
    name: Hashable | None,
    device: jax.Device | jax.sharding.Sharding | None,
) -> object:
    try:
        return jax.device_put(data, device=device)
    except TypeError as error:
        if not _is_payload_conversion_error(data, error):
            raise
        label = f" {name!r}" if name is not None else ""
        raise TypeError(
            f"{container}{label} data cannot be represented by JAX"
        ) from error


def _is_payload_conversion_error(data: object, error: TypeError) -> bool:
    """Identify JAX errors caused by an unsupported data payload."""
    dtype = getattr(data, "dtype", None)
    if dtype is not None:
        try:
            if str(dtype).lower() in {"object", "str", "string", "bytes"}:
                return True
        except (TypeError, ValueError):
            pass
    message = str(error).lower()
    return (
        "not a valid jax array type" in message
        or "only arrays of numeric types are supported by jax" in message
    )


def _with_data(obj: xr.DataArray, data: object) -> xr.DataArray:
    result = obj.copy(deep=False)
    result.data = data
    return result


__all__ = ["device_get", "device_put"]
