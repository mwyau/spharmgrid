# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Explicit Xarray accessors for the JAX array API."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Literal, cast

import xarray as xr
from jax import Array

from .._accessor_utils import (
    as_divergent_source as _as_divergent_source,
)
from .._accessor_utils import (
    as_rotational_source as _as_rotational_source,
)
from .._accessor_utils import (
    find_single_source as _find_single_source,
)
from .._accessor_utils import (
    resolve_dataset_wind_source as _resolve_dataset_wind_source,
)
from .._kinematics_types import (
    DivergentWindSource,
    RotationalWindSource,
    ScalarSource,
    WindSource,
)
from .._transform import TransformSpec
from .._xarray import FieldLayout, require_dataarray
from ..grids import Grid, detect_grid
from ..kinematics import _resolve_wind_inputs, _validate_distinct_names
from ..metadata import (
    Quantity,
    find_variable,
    gradient_metadata,
    identify_scalar_source,
    inverse_gradient_metadata,
    operator_metadata,
    output_metadata,
    vector_operator_metadata,
    wind_component_metadata,
)
from ..operators import EARTH_RADIUS_M, _validate_output_name
from ..operators import _validate_component_names as _validate_operator_names
from ..regrid import _validate_component_names as _validate_regrid_names
from ._ops import (
    divergence as calculate_divergence,
)
from ._ops import (
    divergent_wind as calculate_divergent_wind,
)
from ._ops import (
    filter as calculate_filter,
)
from ._ops import (
    gradient as calculate_gradient,
)
from ._ops import (
    helmholtz as calculate_helmholtz,
)
from ._ops import (
    inverse_gradient as calculate_inverse_gradient,
)
from ._ops import (
    inverse_laplacian as calculate_inverse_laplacian,
)
from ._ops import (
    inverse_vector_laplacian as calculate_inverse_vector_laplacian,
)
from ._ops import (
    kinematics as calculate_kinematics,
)
from ._ops import (
    laplacian as calculate_laplacian,
)
from ._ops import (
    potentials as calculate_potentials,
)
from ._ops import (
    regrid as calculate_regrid,
)
from ._ops import (
    regrid_vector as calculate_regrid_vector,
)
from ._ops import (
    rotational_wind as calculate_rotational_wind,
)
from ._ops import (
    streamfunction as calculate_streamfunction,
)
from ._ops import (
    vector_laplacian as calculate_vector_laplacian,
)
from ._ops import (
    velocity_potential as calculate_velocity_potential,
)
from ._ops import (
    vorticity as calculate_vorticity,
)
from ._ops import (
    wind as calculate_wind,
)
from ._xarray import (
    _apply_pair_operation,
    _apply_scalar,
    _apply_scalar_operation,
    _dataset_outputs,
    _require_jax_data,
    _wrap,
)


def _quantity_output(
    data: Array,
    template: xr.DataArray,
    source: FieldLayout,
    target: FieldLayout,
    *,
    quantity: Quantity,
    name: str,
) -> xr.DataArray:
    return _wrap(
        data,
        template,
        source,
        target,
        name=name,
        attrs=output_metadata(quantity),
    )


def _pair_quantity(
    first: xr.DataArray,
    second: xr.DataArray,
    operation: Callable[..., Array],
    *,
    output: str,
    quantity: Quantity,
    radius: float,
) -> xr.DataArray:
    prepared, target, result = _apply_pair_operation(
        first,
        second,
        operation,
        names=("u", "v"),
        radius=radius,
    )
    return _quantity_output(
        result,
        prepared.first,
        prepared.layout,
        target,
        quantity=quantity,
        name=output,
    )


def _scalar_dataset(
    field: xr.DataArray,
    operation: Callable[..., object],
    *,
    names: tuple[str, ...],
    attrs: tuple[Mapping[str, object] | None, ...],
    radius: float,
) -> xr.Dataset:
    prepared, target, result = _apply_scalar_operation(
        field,
        operation,
        radius=radius,
    )
    return _dataset_outputs(
        cast(tuple[Array, ...], result),
        prepared.field,
        prepared.layout,
        target,
        names=names,
        attrs=attrs,
    )


def _pair_dataset(
    first: xr.DataArray,
    second: xr.DataArray,
    operation: Callable[..., object],
    *,
    names: tuple[str, ...],
    attrs: tuple[Mapping[str, object] | None, ...],
    distinct_names: bool = False,
    radius: float,
) -> xr.Dataset:
    if distinct_names:
        _validate_distinct_names(*names)
    prepared, target, result = _apply_pair_operation(
        first,
        second,
        operation,
        names=("u", "v"),
        radius=radius,
    )
    return _dataset_outputs(
        cast(tuple[Array, ...], result),
        prepared.first,
        prepared.layout,
        target,
        names=names,
        attrs=attrs,
    )


def _single_source_wind(
    field: xr.DataArray,
    *,
    source: ScalarSource | None,
    kind: Literal["rotational", "divergent"],
    eastward: str,
    northward: str,
    radius: float,
) -> xr.Dataset:
    _validate_distinct_names(eastward, northward)
    field = require_dataarray(field)
    _require_jax_data(field, "field")
    if kind == "rotational":
        operation = calculate_rotational_wind
        resolved = _as_rotational_source(
            identify_scalar_source(
                field,
                allowed=("vorticity", "streamfunction"),
                source=source,
            )
        )
    else:
        operation = calculate_divergent_wind
        resolved = _as_divergent_source(
            identify_scalar_source(
                field,
                allowed=("divergence", "velocity_potential"),
                source=source,
            )
        )
    prepared, target, result = _apply_scalar_operation(
        field,
        operation,
        source=cast(ScalarSource, resolved),
        radius=radius,
    )
    output_u, output_v = result
    return _dataset_outputs(
        (output_u, output_v),
        prepared.field,
        prepared.layout,
        target,
        names=(eastward, northward),
        attrs=(
            wind_component_metadata("eastward", kind),
            wind_component_metadata("northward", kind),
        ),
    )


def _vector_operator(
    u: xr.DataArray,
    v: xr.DataArray,
    *,
    inverse: bool,
    eastward: str,
    northward: str,
    radius: float,
) -> xr.Dataset:
    _validate_distinct_names(eastward, northward)
    operation = (
        calculate_inverse_vector_laplacian if inverse else calculate_vector_laplacian
    )
    operation_name = "inverse_laplacian" if inverse else "laplacian"
    prepared, target, result = _apply_pair_operation(
        u,
        v,
        operation,
        names=("u", "v"),
        radius=radius,
    )
    output_u, output_v = result
    return _dataset_outputs(
        (output_u, output_v),
        prepared.first,
        prepared.layout,
        target,
        names=(eastward, northward),
        attrs=(
            vector_operator_metadata(prepared.first, "eastward", operation_name),
            vector_operator_metadata(prepared.second, "northward", operation_name),
        ),
    )


def _regrid_vector(
    u: xr.DataArray,
    v: xr.DataArray,
    target_grid: Grid | xr.DataArray | xr.Dataset,
    truncation: str | TransformSpec | None = None,
    *,
    lmin: int | None = None,
    lmax: int | None = None,
    taper: float | None = None,
    eastward: str = "u",
    northward: str = "v",
) -> xr.Dataset:
    _validate_regrid_names(eastward, northward)
    prepared, target, result = _apply_pair_operation(
        u,
        v,
        calculate_regrid_vector,
        truncation,
        names=("u", "v"),
        target=target_grid,
        lmin=lmin,
        lmax=lmax,
        taper=taper,
    )
    output_u, output_v = result
    # Both numerical outputs use the first component's restored dimension
    # order. The second component contributes its own variable metadata.
    return _dataset_outputs(
        (output_u, output_v),
        prepared.first,
        prepared.layout,
        target,
        names=(eastward, northward),
        attrs=(prepared.first.attrs, prepared.second.attrs),
    )


def _inverse_gradient(
    eastward: xr.DataArray,
    northward: xr.DataArray,
    *,
    output: str | None,
    radius: float,
) -> xr.DataArray:
    _validate_output_name(output)
    prepared, target, result = _apply_pair_operation(
        eastward,
        northward,
        calculate_inverse_gradient,
        names=("eastward", "northward"),
        radius=radius,
    )
    return _wrap(
        result,
        prepared.first,
        prepared.layout,
        target,
        name=output,
        attrs=inverse_gradient_metadata(prepared.first, prepared.second),
    )


def _wind(
    first: xr.DataArray,
    second: xr.DataArray,
    *,
    source: WindSource | None,
    eastward: str,
    northward: str,
    radius: float,
) -> xr.Dataset:
    _validate_distinct_names(eastward, northward)
    first = require_dataarray(first, "first")
    second = require_dataarray(second, "second")
    _require_jax_data(first, "first")
    _require_jax_data(second, "second")
    resolved_source, scalar_one, scalar_two = _resolve_wind_inputs(
        first,
        second,
        source,
    )
    prepared, target, result = _apply_pair_operation(
        scalar_one,
        scalar_two,
        calculate_wind,
        names=("first", "second"),
        source=resolved_source,
        radius=radius,
    )
    output_u, output_v = result
    return _dataset_outputs(
        (output_u, output_v),
        prepared.first,
        prepared.layout,
        target,
        names=(eastward, northward),
        attrs=(output_metadata("u"), output_metadata("v")),
    )


@xr.register_dataarray_accessor("sgj")
class DataArrayAccessor:
    """Labeled JAX operations for one Xarray field."""

    def __init__(self, xarray_obj: xr.DataArray) -> None:
        self._obj = xarray_obj

    @property
    def grid(self) -> Grid:
        """The detected supported GL or CC horizontal grid."""
        return detect_grid(self._obj)

    @property
    def grid_type(self) -> Literal["gl", "cc"]:
        """The lowercase detected grid family."""
        return self.grid.kind

    def filter(
        self,
        truncation: str | TransformSpec | None = None,
        *,
        lmin: int | None = None,
        lmax: int | None = None,
        taper: float | None = None,
    ) -> xr.DataArray:
        """Filter this JAX-backed field by total wavenumber."""
        return _apply_scalar(
            self._obj,
            calculate_filter,
            truncation,
            name=self._obj.name,
            lmin=lmin,
            lmax=lmax,
            taper=taper,
        )

    def regrid(
        self,
        target_grid: Grid | xr.DataArray | xr.Dataset,
        truncation: str | TransformSpec | None = None,
        *,
        lmin: int | None = None,
        lmax: int | None = None,
        taper: float | None = None,
    ) -> xr.DataArray:
        """Regrid this JAX-backed field, optionally filtering in one cycle."""
        return _apply_scalar(
            self._obj,
            calculate_regrid,
            truncation,
            target=target_grid,
            name=self._obj.name,
            lmin=lmin,
            lmax=lmax,
            taper=taper,
        )

    def regrid_vector(
        self,
        v: xr.DataArray,
        target_grid: Grid | xr.DataArray | xr.Dataset,
        truncation: str | TransformSpec | None = None,
        *,
        lmin: int | None = None,
        lmax: int | None = None,
        taper: float | None = None,
        eastward: str = "u",
        northward: str = "v",
    ) -> xr.Dataset:
        """Regrid this eastward component and v through one vector transform."""
        return _regrid_vector(
            self._obj,
            v,
            target_grid,
            truncation,
            lmin=lmin,
            lmax=lmax,
            taper=taper,
            eastward=eastward,
            northward=northward,
        )

    def gradient(
        self,
        *,
        eastward: str = "gradient_eastward",
        northward: str = "gradient_northward",
        radius: float = EARTH_RADIUS_M,
    ) -> xr.Dataset:
        """Compute this scalar field's physical horizontal gradient."""
        _validate_operator_names(eastward, northward)
        return _scalar_dataset(
            self._obj,
            calculate_gradient,
            names=(eastward, northward),
            attrs=(
                gradient_metadata(self._obj, "eastward"),
                gradient_metadata(self._obj, "northward"),
            ),
            radius=radius,
        )

    def laplacian(self, *, radius: float = EARTH_RADIUS_M) -> xr.DataArray:
        """Compute this field's physical spherical Laplacian."""
        return _apply_scalar(
            self._obj,
            calculate_laplacian,
            name=self._obj.name,
            attrs=operator_metadata(self._obj, "laplacian"),
            radius=radius,
        )

    def inverse_laplacian(self, *, radius: float = EARTH_RADIUS_M) -> xr.DataArray:
        """Compute this field's zero-mean inverse spherical Laplacian."""
        return _apply_scalar(
            self._obj,
            calculate_inverse_laplacian,
            name=self._obj.name,
            attrs=operator_metadata(self._obj, "inverse_laplacian"),
            radius=radius,
        )

    def inverse_gradient(
        self,
        northward: xr.DataArray,
        *,
        output: str | None = None,
        radius: float = EARTH_RADIUS_M,
    ) -> xr.DataArray:
        """Recover a scalar potential from two gradient components."""
        return _inverse_gradient(self._obj, northward, output=output, radius=radius)

    def vorticity(
        self,
        v: xr.DataArray,
        *,
        output: str = "vo",
        radius: float = EARTH_RADIUS_M,
    ) -> xr.DataArray:
        """Treat this field as eastward wind and compute relative vorticity."""
        return _pair_quantity(
            self._obj,
            v,
            calculate_vorticity,
            output=output,
            quantity="vo",
            radius=radius,
        )

    def divergence(
        self,
        v: xr.DataArray,
        *,
        output: str = "d",
        radius: float = EARTH_RADIUS_M,
    ) -> xr.DataArray:
        """Treat this field as eastward wind and compute divergence."""
        return _pair_quantity(
            self._obj,
            v,
            calculate_divergence,
            output=output,
            quantity="d",
            radius=radius,
        )

    def kinematics(
        self,
        v: xr.DataArray,
        *,
        vorticity: str = "vo",
        divergence: str = "d",
        radius: float = EARTH_RADIUS_M,
    ) -> xr.Dataset:
        """Treat this field as eastward wind and compute both diagnostics."""
        return _pair_dataset(
            self._obj,
            v,
            calculate_kinematics,
            names=(vorticity, divergence),
            attrs=(output_metadata("vo"), output_metadata("d")),
            distinct_names=True,
            radius=radius,
        )

    def streamfunction(
        self,
        v: xr.DataArray,
        *,
        output: str = "strf",
        radius: float = EARTH_RADIUS_M,
    ) -> xr.DataArray:
        """Treat this field as eastward wind and calculate streamfunction."""
        return _pair_quantity(
            self._obj,
            v,
            calculate_streamfunction,
            output=output,
            quantity="strf",
            radius=radius,
        )

    def velocity_potential(
        self,
        v: xr.DataArray,
        *,
        output: str = "vp",
        radius: float = EARTH_RADIUS_M,
    ) -> xr.DataArray:
        """Treat this field as eastward wind and calculate velocity potential."""
        return _pair_quantity(
            self._obj,
            v,
            calculate_velocity_potential,
            output=output,
            quantity="vp",
            radius=radius,
        )

    def potentials(
        self,
        v: xr.DataArray,
        *,
        streamfunction: str = "strf",
        velocity_potential: str = "vp",
        radius: float = EARTH_RADIUS_M,
    ) -> xr.Dataset:
        """Treat this field as eastward wind and calculate both potentials."""
        return _pair_dataset(
            self._obj,
            v,
            calculate_potentials,
            names=(streamfunction, velocity_potential),
            attrs=(output_metadata("strf"), output_metadata("vp")),
            distinct_names=True,
            radius=radius,
        )

    def helmholtz(
        self,
        v: xr.DataArray,
        *,
        divergent_eastward: str = "u_divergent",
        divergent_northward: str = "v_divergent",
        rotational_eastward: str = "u_rotational",
        rotational_northward: str = "v_rotational",
        radius: float = EARTH_RADIUS_M,
    ) -> xr.Dataset:
        """Split this wind into divergent and rotational components."""
        return _pair_dataset(
            self._obj,
            v,
            calculate_helmholtz,
            names=(
                divergent_eastward,
                divergent_northward,
                rotational_eastward,
                rotational_northward,
            ),
            attrs=(
                wind_component_metadata("eastward", "divergent"),
                wind_component_metadata("northward", "divergent"),
                wind_component_metadata("eastward", "rotational"),
                wind_component_metadata("northward", "rotational"),
            ),
            distinct_names=True,
            radius=radius,
        )

    def vector_laplacian(
        self,
        v: xr.DataArray,
        *,
        eastward: str = "u",
        northward: str = "v",
        radius: float = EARTH_RADIUS_M,
    ) -> xr.Dataset:
        """Apply the vector Laplacian to this wind field."""
        return _vector_operator(
            self._obj,
            v,
            inverse=False,
            eastward=eastward,
            northward=northward,
            radius=radius,
        )

    def inverse_vector_laplacian(
        self,
        v: xr.DataArray,
        *,
        eastward: str = "u",
        northward: str = "v",
        radius: float = EARTH_RADIUS_M,
    ) -> xr.Dataset:
        """Apply the zero-mode-defined inverse vector Laplacian."""
        return _vector_operator(
            self._obj,
            v,
            inverse=True,
            eastward=eastward,
            northward=northward,
            radius=radius,
        )

    def rotational_wind(
        self,
        *,
        source: RotationalWindSource | None = None,
        eastward: str = "u_rotational",
        northward: str = "v_rotational",
        radius: float = EARTH_RADIUS_M,
    ) -> xr.Dataset:
        """Recover rotational wind from this scalar field."""
        return _single_source_wind(
            self._obj,
            source=source,
            kind="rotational",
            eastward=eastward,
            northward=northward,
            radius=radius,
        )

    def divergent_wind(
        self,
        *,
        source: DivergentWindSource | None = None,
        eastward: str = "u_divergent",
        northward: str = "v_divergent",
        radius: float = EARTH_RADIUS_M,
    ) -> xr.Dataset:
        """Recover divergent wind from this scalar field."""
        return _single_source_wind(
            self._obj,
            source=source,
            kind="divergent",
            eastward=eastward,
            northward=northward,
            radius=radius,
        )

    def wind(
        self,
        second: xr.DataArray,
        *,
        source: WindSource | None = None,
        eastward: str = "u",
        northward: str = "v",
        radius: float = EARTH_RADIUS_M,
    ) -> xr.Dataset:
        """Reconstruct wind from this field and a second scalar source."""
        return _wind(
            self._obj,
            second,
            source=source,
            eastward=eastward,
            northward=northward,
            radius=radius,
        )


@xr.register_dataset_accessor("sgj")
class DatasetAccessor:
    """CF-aware JAX operations for an Xarray Dataset."""

    def __init__(self, xarray_obj: xr.Dataset) -> None:
        self._obj = xarray_obj

    @property
    def grid(self) -> Grid:
        """The Dataset's detected supported global grid."""
        return detect_grid(self._obj)

    @property
    def grid_type(self) -> Literal["gl", "cc"]:
        """The lowercase detected grid family."""
        return self.grid.kind

    def regrid_vector(
        self,
        target_grid: Grid | xr.DataArray | xr.Dataset,
        truncation: str | TransformSpec | None = None,
        *,
        u: str | None = None,
        v: str | None = None,
        lmin: int | None = None,
        lmax: int | None = None,
        taper: float | None = None,
        eastward: str = "u",
        northward: str = "v",
    ) -> xr.Dataset:
        """Find wind components and regrid them as one vector field."""
        return _regrid_vector(
            find_variable(self._obj, "u", u),
            find_variable(self._obj, "v", v),
            target_grid,
            truncation,
            lmin=lmin,
            lmax=lmax,
            taper=taper,
            eastward=eastward,
            northward=northward,
        )

    def inverse_gradient(
        self,
        *,
        eastward: str = "gradient_eastward",
        northward: str = "gradient_northward",
        output: str | None = None,
        radius: float = EARTH_RADIUS_M,
    ) -> xr.DataArray:
        """Recover a scalar potential from named gradient components."""
        if eastward not in self._obj.data_vars or northward not in self._obj.data_vars:
            raise ValueError(
                "inverse_gradient requires explicit gradient component variables; "
                "pass eastward= and northward="
            )
        return _inverse_gradient(
            self._obj[eastward],
            self._obj[northward],
            output=output,
            radius=radius,
        )

    def vorticity(
        self,
        *,
        u: str | None = None,
        v: str | None = None,
        output: str = "vo",
        radius: float = EARTH_RADIUS_M,
    ) -> xr.DataArray:
        """Find wind components and compute relative vorticity."""
        return _pair_quantity(
            find_variable(self._obj, "u", u),
            find_variable(self._obj, "v", v),
            calculate_vorticity,
            output=output,
            quantity="vo",
            radius=radius,
        )

    def divergence(
        self,
        *,
        u: str | None = None,
        v: str | None = None,
        output: str = "d",
        radius: float = EARTH_RADIUS_M,
    ) -> xr.DataArray:
        """Find wind components and compute divergence."""
        return _pair_quantity(
            find_variable(self._obj, "u", u),
            find_variable(self._obj, "v", v),
            calculate_divergence,
            output=output,
            quantity="d",
            radius=radius,
        )

    def kinematics(
        self,
        *,
        u: str | None = None,
        v: str | None = None,
        vorticity: str = "vo",
        divergence: str = "d",
        radius: float = EARTH_RADIUS_M,
    ) -> xr.Dataset:
        """Find wind components and compute vorticity plus divergence."""
        return _pair_dataset(
            find_variable(self._obj, "u", u),
            find_variable(self._obj, "v", v),
            calculate_kinematics,
            names=(vorticity, divergence),
            attrs=(output_metadata("vo"), output_metadata("d")),
            distinct_names=True,
            radius=radius,
        )

    def streamfunction(
        self,
        *,
        u: str | None = None,
        v: str | None = None,
        output: str = "strf",
        radius: float = EARTH_RADIUS_M,
    ) -> xr.DataArray:
        """Find wind components and compute streamfunction."""
        return _pair_quantity(
            find_variable(self._obj, "u", u),
            find_variable(self._obj, "v", v),
            calculate_streamfunction,
            output=output,
            quantity="strf",
            radius=radius,
        )

    def velocity_potential(
        self,
        *,
        u: str | None = None,
        v: str | None = None,
        output: str = "vp",
        radius: float = EARTH_RADIUS_M,
    ) -> xr.DataArray:
        """Find wind components and compute velocity potential."""
        return _pair_quantity(
            find_variable(self._obj, "u", u),
            find_variable(self._obj, "v", v),
            calculate_velocity_potential,
            output=output,
            quantity="vp",
            radius=radius,
        )

    def potentials(
        self,
        *,
        u: str | None = None,
        v: str | None = None,
        streamfunction: str = "strf",
        velocity_potential: str = "vp",
        radius: float = EARTH_RADIUS_M,
    ) -> xr.Dataset:
        """Find wind components and compute both scalar potentials."""
        return _pair_dataset(
            find_variable(self._obj, "u", u),
            find_variable(self._obj, "v", v),
            calculate_potentials,
            names=(streamfunction, velocity_potential),
            attrs=(output_metadata("strf"), output_metadata("vp")),
            distinct_names=True,
            radius=radius,
        )

    def helmholtz(
        self,
        *,
        u: str | None = None,
        v: str | None = None,
        divergent_eastward: str = "u_divergent",
        divergent_northward: str = "v_divergent",
        rotational_eastward: str = "u_rotational",
        rotational_northward: str = "v_rotational",
        radius: float = EARTH_RADIUS_M,
    ) -> xr.Dataset:
        """Find wind components and split them into divergent and rotational wind."""
        return _pair_dataset(
            find_variable(self._obj, "u", u),
            find_variable(self._obj, "v", v),
            calculate_helmholtz,
            names=(
                divergent_eastward,
                divergent_northward,
                rotational_eastward,
                rotational_northward,
            ),
            attrs=(
                wind_component_metadata("eastward", "divergent"),
                wind_component_metadata("northward", "divergent"),
                wind_component_metadata("eastward", "rotational"),
                wind_component_metadata("northward", "rotational"),
            ),
            distinct_names=True,
            radius=radius,
        )

    def vector_laplacian(
        self,
        *,
        u: str | None = None,
        v: str | None = None,
        eastward: str = "u",
        northward: str = "v",
        radius: float = EARTH_RADIUS_M,
    ) -> xr.Dataset:
        """Find wind components and apply the vector Laplacian."""
        return _vector_operator(
            find_variable(self._obj, "u", u),
            find_variable(self._obj, "v", v),
            inverse=False,
            eastward=eastward,
            northward=northward,
            radius=radius,
        )

    def inverse_vector_laplacian(
        self,
        *,
        u: str | None = None,
        v: str | None = None,
        eastward: str = "u",
        northward: str = "v",
        radius: float = EARTH_RADIUS_M,
    ) -> xr.Dataset:
        """Find wind components and apply the inverse vector Laplacian."""
        return _vector_operator(
            find_variable(self._obj, "u", u),
            find_variable(self._obj, "v", v),
            inverse=True,
            eastward=eastward,
            northward=northward,
            radius=radius,
        )

    def rotational_wind(
        self,
        *,
        field: str | None = None,
        source: RotationalWindSource | None = None,
        eastward: str = "u_rotational",
        northward: str = "v_rotational",
        radius: float = EARTH_RADIUS_M,
    ) -> xr.Dataset:
        """Find a scalar source and recover rotational wind."""
        selected, inferred = _find_single_source(
            self._obj,
            field,
            source,
            primary="vo",
            secondary="strf",
        )
        return _single_source_wind(
            selected,
            source=_as_rotational_source(inferred),
            kind="rotational",
            eastward=eastward,
            northward=northward,
            radius=radius,
        )

    def divergent_wind(
        self,
        *,
        field: str | None = None,
        source: DivergentWindSource | None = None,
        eastward: str = "u_divergent",
        northward: str = "v_divergent",
        radius: float = EARTH_RADIUS_M,
    ) -> xr.Dataset:
        """Find a scalar source and recover divergent wind."""
        selected, inferred = _find_single_source(
            self._obj,
            field,
            source,
            primary="d",
            secondary="vp",
        )
        return _single_source_wind(
            selected,
            source=_as_divergent_source(inferred),
            kind="divergent",
            eastward=eastward,
            northward=northward,
            radius=radius,
        )

    def wind(
        self,
        *,
        source: WindSource | None = None,
        vorticity: str | None = None,
        divergence: str | None = None,
        streamfunction: str | None = None,
        velocity_potential: str | None = None,
        eastward: str = "u",
        northward: str = "v",
        radius: float = EARTH_RADIUS_M,
    ) -> xr.Dataset:
        """Reconstruct wind from one complete scalar representation."""
        resolved_source, first, second = _resolve_dataset_wind_source(
            self._obj,
            source=source,
            vorticity=vorticity,
            divergence=divergence,
            streamfunction=streamfunction,
            velocity_potential=velocity_potential,
        )
        return _wind(
            first,
            second,
            source=resolved_source,
            eastward=eastward,
            northward=northward,
            radius=radius,
        )


__all__ = ["DataArrayAccessor", "DatasetAccessor"]
