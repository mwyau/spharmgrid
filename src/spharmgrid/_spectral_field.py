# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reusable analyzed scalar and vector fields for the Xarray API."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any, cast

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
    vector_analysis,
    vector_synthesis,
)
from ._transform import TransformSpec
from ._vector import vector_inputs
from ._xarray import (
    FieldLayout,
    apply_ufunc_options,
    field_layout,
    require_dataarray,
    restore_output,
    target_layout,
)
from .grids import Grid
from .metadata import (
    operator_metadata,
    output_metadata,
    vector_operator_metadata,
    wind_component_metadata,
)
from .spectral import (
    _degree_scale,
    _intersect_transform_spec,
    _inverse_laplacian_multiplier,
    _laplacian_multiplier,
    _resolve_spectral_spec,
    _spectral_selection_is_within,
    _spectral_selection_weights,
    _validate_taper,
    resolve_transform_spec,
)

_MODE_DIM_BASE = "__spharmgrid_spectral_mode__"
_COMPONENT_DIM_BASE = "__spharmgrid_vector_component__"


def analyze(
    field: xr.DataArray,
    truncation: str | TransformSpec | None = None,
    *,
    lmin: int | None = None,
    lmax: int | None = None,
    sht_threads: int | None = None,
) -> SpectralField:
    """Analyze one Xarray scalar field into a reusable spectral field."""
    field = require_dataarray(field)
    layout = field_layout(field)
    selection = _resolve_spectral_spec(truncation, lmin=lmin, lmax=lmax)
    spec = resolve_transform_spec(layout.grid, layout.grid, selection)
    nthreads = resolve_sht_threads(sht_threads, dask=field.chunks is not None)
    return _analyze_scalar(field, layout, spec, selection, nthreads)


def analyze_vector(
    u: xr.DataArray,
    v: xr.DataArray,
    truncation: str | TransformSpec | None = None,
    *,
    lmin: int | None = None,
    lmax: int | None = None,
    sht_threads: int | None = None,
) -> SpectralVectorField:
    """Analyze one geographic vector field into reusable E/B coefficients."""
    layout, canonical_u, canonical_v = vector_inputs(u, v)
    selection = _resolve_spectral_spec(truncation, lmin=lmin, lmax=lmax)
    spec = resolve_transform_spec(layout.grid, layout.grid, selection)
    if spec.lmax < 1:
        raise ValueError("vector analysis requires a grid supporting total degree l=1")
    dask = canonical_u.chunks is not None or canonical_v.chunks is not None
    nthreads = resolve_sht_threads(sht_threads, dask=dask)
    return _analyze_vector(
        u,
        v,
        layout,
        canonical_u,
        canonical_v,
        spec,
        selection,
        nthreads,
    )


@dataclass(frozen=True, slots=True)
class SpectralField:
    """An analyzed scalar field with private DUCC coefficient storage."""

    _coefficients: xr.DataArray
    _spec: TransformSpec
    _layout: FieldLayout
    _original_dims: tuple[Hashable, ...]
    _mode_dim: str
    _nthreads: int
    _name: Hashable | None
    _attrs: dict[str, Any]
    _synthesis: Callable[..., Any] = scalar_synthesis

    @property
    def grid(self) -> Grid:
        """The source grid used for analysis."""
        return self._layout.grid

    @property
    def spec(self) -> TransformSpec:
        """The private transform domain's descriptive limits."""
        return self._spec

    def filter(
        self,
        truncation: str | TransformSpec | None = None,
        *,
        lmin: int | None = None,
        lmax: int | None = None,
        taper: float | None = None,
    ) -> SpectralField:
        """Apply a spectral selection without synthesizing the field."""
        selection = _resolve_spectral_spec(truncation, lmin=lmin, lmax=lmax)
        _validate_taper(taper)
        if selection is None and taper is None:
            return self
        _validate_selection(self._spec, selection)
        if selection == self._spec and taper is None:
            return self
        coefficients = _apply_weights(
            self._coefficients,
            self._mode_dim,
            self._spec,
            selection,
            taper,
        )
        return _replace_scalar(self, coefficients)

    def laplacian(self, *, radius: float = 6_371_220.0) -> SpectralField:
        """Apply the spherical Laplacian to the stored coefficients."""
        _validate_radius(radius)
        multiplier = _laplacian_multiplier(self._spec, radius)
        coefficients = _multiply_mode(self._coefficients, self._mode_dim, multiplier)
        source = _metadata_source(self._name, self._attrs)
        return _replace_scalar(
            self,
            coefficients,
            attrs=operator_metadata(source, "laplacian"),
        )

    def inverse_laplacian(self, *, radius: float = 6_371_220.0) -> SpectralField:
        """Apply the zero-mode-defined inverse Laplacian to coefficients."""
        _validate_radius(radius)
        multiplier = _inverse_laplacian_multiplier(self._spec, radius)
        coefficients = _multiply_mode(self._coefficients, self._mode_dim, multiplier)
        source = _metadata_source(self._name, self._attrs)
        return _replace_scalar(
            self,
            coefficients,
            attrs=operator_metadata(source, "inverse_laplacian"),
        )

    def synthesize(self) -> xr.DataArray:
        """Synthesize the representation on its source grid."""
        result = _synthesize_scalar(
            self._coefficients,
            self._mode_dim,
            self._spec,
            self._layout,
            self._layout,
            self._original_dims,
            self._nthreads,
            self._synthesis,
        )
        return _set_scalar_metadata(result, self._name, self._attrs)

    def regrid(
        self,
        target: Grid | xr.DataArray | xr.Dataset,
        truncation: str | TransformSpec | None = None,
        *,
        lmin: int | None = None,
        lmax: int | None = None,
        taper: float | None = None,
    ) -> xr.DataArray:
        """Synthesize the representation on another supported grid."""
        selection = _resolve_spectral_spec(truncation, lmin=lmin, lmax=lmax)
        _validate_taper(taper)
        effective = self._spec if selection is None else selection
        _validate_selection(self._spec, effective)
        target_description = target_layout(target, self._layout)
        if selection is None:
            target_spec = _intersect_transform_spec(
                self._layout.grid, target_description.grid, self._spec
            )
        else:
            target_spec = resolve_transform_spec(
                self._layout.grid, target_description.grid, selection
            )
        coefficients = self._coefficients
        if selection is None and taper is not None:
            coefficients = _apply_weights(
                coefficients,
                self._mode_dim,
                self._spec,
                self._spec,
                taper,
            )
        coefficients = _repack_coefficients(
            coefficients,
            self._mode_dim,
            self._spec,
            target_spec,
        )
        if selection is not None and (
            taper is not None or effective != self._spec or target_spec != self._spec
        ):
            coefficients = _apply_weights(
                coefficients,
                self._mode_dim,
                target_spec,
                effective,
                taper,
            )
        result = _synthesize_scalar(
            coefficients,
            self._mode_dim,
            target_spec,
            self._layout,
            target_description,
            self._original_dims,
            self._nthreads,
            self._synthesis,
        )
        return _set_scalar_metadata(result, self._name, self._attrs)


@dataclass(frozen=True, slots=True)
class SpectralVectorField:
    """An analyzed geographic vector field with private DUCC E/B storage."""

    _coefficients: xr.DataArray
    _spec: TransformSpec
    _layout: FieldLayout
    _original_dims: tuple[Hashable, ...]
    _component_dim: str
    _mode_dim: str
    _nthreads: int
    _u_name: Hashable | None
    _v_name: Hashable | None
    _u_attrs: dict[str, Any]
    _v_attrs: dict[str, Any]
    _scalar_synthesis: Callable[..., Any] = scalar_synthesis
    _vector_synthesis: Callable[..., Any] = vector_synthesis

    @property
    def grid(self) -> Grid:
        """The source grid used for vector analysis."""
        return self._layout.grid

    @property
    def spec(self) -> TransformSpec:
        """The private vector transform domain's descriptive limits."""
        return self._spec

    def filter(
        self,
        truncation: str | TransformSpec | None = None,
        *,
        lmin: int | None = None,
        lmax: int | None = None,
        taper: float | None = None,
    ) -> SpectralVectorField:
        """Apply a spectral selection without synthesizing the vector."""
        selection = _resolve_spectral_spec(truncation, lmin=lmin, lmax=lmax)
        _validate_taper(taper)
        if selection is None and taper is None:
            return self
        _validate_selection(self._spec, selection)
        if selection == self._spec and taper is None:
            return self
        coefficients = _apply_weights(
            self._coefficients,
            self._mode_dim,
            self._spec,
            selection,
            taper,
        )
        return _replace_vector(self, coefficients)

    def laplacian(self, *, radius: float = 6_371_220.0) -> SpectralVectorField:
        """Apply the vector spherical Laplacian to E/B coefficients."""
        return self._laplacian(radius=radius, inverse=False)

    def inverse_laplacian(self, *, radius: float = 6_371_220.0) -> SpectralVectorField:
        """Apply the zero-mode-defined inverse vector Laplacian."""
        return self._laplacian(radius=radius, inverse=True)

    def vorticity(self, *, radius: float = 6_371_220.0) -> SpectralField:
        """Derive relative-vorticity coefficients from the stored vector."""
        _validate_radius(radius)
        scale = _degree_scale(self._spec, radius)
        coefficients = -self._coefficients.sel(
            {self._component_dim: 1}, drop=True
        ) * xr.DataArray(scale, dims=(self._mode_dim,))
        return _scalar_from_vector(self, coefficients, "vo", output_metadata("vo"))

    def divergence(self, *, radius: float = 6_371_220.0) -> SpectralField:
        """Derive divergence coefficients from the stored vector."""
        _validate_radius(radius)
        scale = _degree_scale(self._spec, radius)
        coefficients = -self._coefficients.sel(
            {self._component_dim: 0}, drop=True
        ) * xr.DataArray(scale, dims=(self._mode_dim,))
        return _scalar_from_vector(self, coefficients, "d", output_metadata("d"))

    def streamfunction(self, *, radius: float = 6_371_220.0) -> SpectralField:
        """Derive streamfunction coefficients from the stored vector."""
        return self._potential("strf", radius)

    def velocity_potential(self, *, radius: float = 6_371_220.0) -> SpectralField:
        """Derive velocity-potential coefficients from the stored vector."""
        return self._potential("vp", radius)

    def divergent(self) -> SpectralVectorField:
        """Keep only the divergent E component."""
        zeros = xr.zeros_like(self._coefficients.sel({self._component_dim: 1}))
        coefficients = xr.concat(
            (self._coefficients.sel({self._component_dim: 0}), zeros),
            dim=self._component_dim,
        ).assign_coords({self._component_dim: [0, 1]})
        return _replace_vector(
            self,
            coefficients,
            names=("u_divergent", "v_divergent"),
            attrs=(
                wind_component_metadata("eastward", "divergent"),
                wind_component_metadata("northward", "divergent"),
            ),
        )

    def rotational(self) -> SpectralVectorField:
        """Keep only the rotational B component."""
        zeros = xr.zeros_like(self._coefficients.sel({self._component_dim: 0}))
        coefficients = xr.concat(
            (zeros, self._coefficients.sel({self._component_dim: 1})),
            dim=self._component_dim,
        ).assign_coords({self._component_dim: [0, 1]})
        return _replace_vector(
            self,
            coefficients,
            names=("u_rotational", "v_rotational"),
            attrs=(
                wind_component_metadata("eastward", "rotational"),
                wind_component_metadata("northward", "rotational"),
            ),
        )

    def synthesize(self) -> tuple[xr.DataArray, xr.DataArray]:
        """Synthesize eastward and northward components on the source grid."""
        return _synthesize_vector(
            self._coefficients,
            self._component_dim,
            self._mode_dim,
            self._spec,
            self._layout,
            self._layout,
            self._original_dims,
            self._nthreads,
            (self._u_name, self._v_name),
            (self._u_attrs, self._v_attrs),
            self._vector_synthesis,
        )

    def regrid(
        self,
        target: Grid | xr.DataArray | xr.Dataset,
        truncation: str | TransformSpec | None = None,
        *,
        lmin: int | None = None,
        lmax: int | None = None,
        taper: float | None = None,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Synthesize the vector representation on another supported grid."""
        selection = _resolve_spectral_spec(truncation, lmin=lmin, lmax=lmax)
        _validate_taper(taper)
        effective = self._spec if selection is None else selection
        _validate_selection(self._spec, effective)
        target_description = target_layout(target, self._layout)
        if selection is None:
            target_spec = _intersect_transform_spec(
                self._layout.grid, target_description.grid, self._spec
            )
        else:
            target_spec = resolve_transform_spec(
                self._layout.grid, target_description.grid, selection
            )
        if target_spec.lmax < 1:
            raise ValueError(
                "vector regridding requires a grid supporting total degree l=1"
            )
        coefficients = self._coefficients
        if selection is None and taper is not None:
            coefficients = _apply_weights(
                coefficients,
                self._mode_dim,
                self._spec,
                self._spec,
                taper,
            )
        coefficients = _repack_coefficients(
            coefficients,
            self._mode_dim,
            self._spec,
            target_spec,
        )
        if selection is not None and (
            taper is not None or effective != self._spec or target_spec != self._spec
        ):
            coefficients = _apply_weights(
                coefficients,
                self._mode_dim,
                target_spec,
                effective,
                taper,
            )
        return _synthesize_vector(
            coefficients,
            self._component_dim,
            self._mode_dim,
            target_spec,
            self._layout,
            target_description,
            self._original_dims,
            self._nthreads,
            (self._u_name, self._v_name),
            (self._u_attrs, self._v_attrs),
            self._vector_synthesis,
        )

    def _laplacian(self, *, radius: float, inverse: bool) -> SpectralVectorField:
        _validate_radius(radius)
        multiplier = (
            _inverse_laplacian_multiplier(self._spec, radius)
            if inverse
            else _laplacian_multiplier(self._spec, radius)
        )
        coefficients = _multiply_mode(self._coefficients, self._mode_dim, multiplier)
        operation = "inverse_laplacian" if inverse else "laplacian"
        attrs = (
            vector_operator_metadata(
                _metadata_source(self._u_name, self._u_attrs),
                "eastward",
                operation,
            ),
            vector_operator_metadata(
                _metadata_source(self._v_name, self._v_attrs),
                "northward",
                operation,
            ),
        )
        return _replace_vector(self, coefficients, attrs=attrs)

    def _potential(self, quantity: str, radius: float) -> SpectralField:
        _validate_radius(radius)
        scale = _degree_scale(self._spec, radius)
        inverse = _inverse_laplacian_multiplier(self._spec, radius)
        component = 1 if quantity == "strf" else 0
        coefficients = -self._coefficients.sel(
            {self._component_dim: component}, drop=True
        ) * xr.DataArray(scale * inverse, dims=(self._mode_dim,))
        return _scalar_from_vector(
            self,
            coefficients,
            quantity,
            output_metadata(cast(Any, quantity)),
        )


def _analyze_scalar(
    field: xr.DataArray,
    layout: FieldLayout,
    spec: TransformSpec,
    selection: TransformSpec | None,
    nthreads: int,
    analysis: Callable[..., Any] = scalar_analysis,
    synthesis: Callable[..., Any] = scalar_synthesis,
) -> SpectralField:
    canonical = layout.canonicalize(field)
    mode_dim = _new_dimension(_MODE_DIM_BASE, canonical.dims)
    mode_size = alm_degrees(spec.lmax, spec.mmax).size

    def kernel(frame: NDArray[np.generic]) -> NDArray[np.complex128]:
        coefficients = analysis(
            frame,
            spec=spec,
            geometry=geometry_for(layout.grid),
            phi0=layout.transform_layout.phi0_radians,
            nthreads=nthreads,
        )
        return coefficients[0]

    coefficients = xr.apply_ufunc(
        kernel,
        canonical,
        input_core_dims=[[layout.latitude_dim, layout.longitude_dim]],
        output_core_dims=[[mode_dim]],
        vectorize=True,
        output_dtypes=[np.complex128],
        **apply_ufunc_options(canonical, output_sizes={mode_dim: mode_size}),
    )
    coefficients = cast(xr.DataArray, coefficients)
    if selection is not None:
        coefficients = _apply_weights(
            coefficients, mode_dim, spec, selection, taper=None
        )
    return SpectralField(
        coefficients,
        spec,
        layout,
        field.dims,
        mode_dim,
        nthreads,
        field.name,
        dict(field.attrs),
        synthesis,
    )


def _analyze_vector(
    u: xr.DataArray,
    v: xr.DataArray,
    layout: FieldLayout,
    canonical_u: xr.DataArray,
    canonical_v: xr.DataArray,
    spec: TransformSpec,
    selection: TransformSpec | None,
    nthreads: int,
    scalar_synthesis_fn: Callable[..., Any] = scalar_synthesis,
    vector_synthesis_fn: Callable[..., Any] = vector_synthesis,
) -> SpectralVectorField:
    component_dim = _new_dimension(_COMPONENT_DIM_BASE, canonical_u.dims)
    mode_dim = _new_dimension(_MODE_DIM_BASE, canonical_u.dims + (component_dim,))
    mode_size = alm_degrees(spec.lmax, spec.mmax).size

    def kernel(
        frame_u: NDArray[np.generic], frame_v: NDArray[np.generic]
    ) -> NDArray[np.complex128]:
        return vector_analysis(
            frame_u,
            frame_v,
            spec=spec,
            geometry=geometry_for(layout.grid),
            phi0=layout.transform_layout.phi0_radians,
            nthreads=nthreads,
        )

    dask_field = canonical_u if canonical_u.chunks is not None else canonical_v
    coefficients = xr.apply_ufunc(
        kernel,
        canonical_u,
        canonical_v,
        input_core_dims=[
            [layout.latitude_dim, layout.longitude_dim],
            [layout.latitude_dim, layout.longitude_dim],
        ],
        output_core_dims=[[component_dim, mode_dim]],
        vectorize=True,
        output_dtypes=[np.complex128],
        **apply_ufunc_options(
            dask_field,
            output_sizes={component_dim: 2, mode_dim: mode_size},
        ),
    )
    coefficients = cast(xr.DataArray, coefficients).assign_coords(
        {component_dim: [0, 1]}
    )
    if selection is not None:
        coefficients = _apply_weights(
            coefficients, mode_dim, spec, selection, taper=None
        )
    return SpectralVectorField(
        coefficients,
        spec,
        layout,
        u.dims,
        component_dim,
        mode_dim,
        nthreads,
        u.name,
        v.name,
        dict(u.attrs),
        dict(v.attrs),
        scalar_synthesis_fn,
        vector_synthesis_fn,
    )


def _synthesize_scalar(
    coefficients: xr.DataArray,
    mode_dim: str,
    spec: TransformSpec,
    source: FieldLayout,
    target: FieldLayout,
    original_dims: tuple[Hashable, ...],
    nthreads: int,
    synthesis: Callable[..., Any] = scalar_synthesis,
) -> xr.DataArray:
    def kernel(frame: NDArray[np.generic]) -> NDArray[np.float64]:
        return synthesis(
            frame[np.newaxis, :],
            spec=spec,
            geometry=geometry_for(target.grid),
            ntheta=target.grid.nlat,
            nphi=target.grid.nlon,
            phi0=target.transform_layout.phi0_radians,
            nthreads=nthreads,
        )

    output_sizes = {
        target.latitude_dim: target.grid.nlat,
        target.longitude_dim: target.grid.nlon,
    }
    result = xr.apply_ufunc(
        kernel,
        coefficients,
        input_core_dims=[[mode_dim]],
        output_core_dims=[[target.latitude_dim, target.longitude_dim]],
        vectorize=True,
        output_dtypes=[np.float64],
        **apply_ufunc_options(coefficients, output_sizes=output_sizes),
    )
    return restore_output(
        cast(xr.DataArray, result),
        source=source,
        target=target,
        original_dims=original_dims,
    )


def _synthesize_vector(
    coefficients: xr.DataArray,
    component_dim: str,
    mode_dim: str,
    spec: TransformSpec,
    source: FieldLayout,
    target: FieldLayout,
    original_dims: tuple[Hashable, ...],
    nthreads: int,
    names: tuple[Hashable | None, Hashable | None],
    attrs: tuple[dict[str, Any], dict[str, Any]],
    synthesis: Callable[..., Any] = vector_synthesis,
) -> tuple[xr.DataArray, xr.DataArray]:
    def kernel(
        frame: NDArray[np.generic],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return synthesis(
            frame[0],
            frame[1],
            spec=spec,
            geometry=geometry_for(target.grid),
            ntheta=target.grid.nlat,
            nphi=target.grid.nlon,
            phi0=target.transform_layout.phi0_radians,
            nthreads=nthreads,
        )

    output_sizes = {
        target.latitude_dim: target.grid.nlat,
        target.longitude_dim: target.grid.nlon,
    }
    results = xr.apply_ufunc(
        kernel,
        coefficients,
        input_core_dims=[[component_dim, mode_dim]],
        output_core_dims=[
            [target.latitude_dim, target.longitude_dim],
            [target.latitude_dim, target.longitude_dim],
        ],
        vectorize=True,
        output_dtypes=[np.float64, np.float64],
        **apply_ufunc_options(coefficients, output_sizes=output_sizes),
    )
    eastward, northward = cast(tuple[xr.DataArray, xr.DataArray], results)
    eastward = restore_output(
        eastward, source=source, target=target, original_dims=original_dims
    )
    northward = restore_output(
        northward, source=source, target=target, original_dims=original_dims
    )
    eastward = eastward.copy(deep=False)
    northward = northward.copy(deep=False)
    eastward.name, northward.name = names
    eastward.attrs, northward.attrs = dict(attrs[0]), dict(attrs[1])
    return eastward, northward


def _repack_coefficients(
    coefficients: xr.DataArray,
    mode_dim: str,
    source: TransformSpec,
    target: TransformSpec,
) -> xr.DataArray:
    if target == source or (target.lmax == source.lmax and target.mmax == source.mmax):
        # The packed DUCC layout is unchanged.  A later explicit selection
        # still applies its logical lmin/rhomboidal mask when needed.
        return coefficients
    if target.lmax <= source.lmax and target.mmax <= source.mmax:
        from ._ducc import alm_subselection

        indices = alm_subselection(source.lmax, source.mmax, target.lmax, target.mmax)
        return coefficients.isel({mode_dim: indices})
    source_degrees = alm_degrees(source.lmax, source.mmax)
    source_orders = alm_orders(source.lmax, source.mmax)
    source_lookup = {
        (int(degree), int(order)): index
        for index, (degree, order) in enumerate(
            zip(source_degrees, source_orders, strict=True)
        )
    }
    target_degrees = alm_degrees(target.lmax, target.mmax)
    target_orders = alm_orders(target.lmax, target.mmax)
    parts: list[xr.DataArray] = []
    for position, (degree, order) in enumerate(
        zip(target_degrees, target_orders, strict=True)
    ):
        source_index = source_lookup.get((int(degree), int(order)))
        if source_index is None:
            part = xr.zeros_like(coefficients.isel({mode_dim: 0}, drop=True))
        else:
            part = coefficients.isel({mode_dim: source_index}, drop=True)
        parts.append(part.expand_dims({mode_dim: [position]}))
    return xr.concat(parts, dim=mode_dim).transpose(
        *[dimension for dimension in coefficients.dims if dimension != mode_dim],
        mode_dim,
    )


def _apply_weights(
    coefficients: xr.DataArray,
    mode_dim: str,
    coefficient_spec: TransformSpec,
    selection: TransformSpec | None,
    taper: float | None,
) -> xr.DataArray:
    weights = _spectral_selection_weights(coefficient_spec, selection, taper)
    return coefficients * xr.DataArray(weights, dims=(mode_dim,))


def _multiply_mode(
    coefficients: xr.DataArray, mode_dim: str, multiplier: NDArray[np.float64]
) -> xr.DataArray:
    return coefficients * xr.DataArray(multiplier, dims=(mode_dim,))


def _scalar_from_vector(
    vector: SpectralVectorField,
    coefficients: xr.DataArray,
    name: str,
    attrs: dict[str, Any],
) -> SpectralField:
    return SpectralField(
        coefficients,
        vector._spec,
        vector._layout,
        vector._original_dims,
        vector._mode_dim,
        vector._nthreads,
        name,
        attrs,
        vector._scalar_synthesis,
    )


def _replace_scalar(
    field: SpectralField,
    coefficients: xr.DataArray,
    *,
    name: Hashable | None = None,
    attrs: dict[str, Any] | None = None,
) -> SpectralField:
    return SpectralField(
        coefficients,
        field._spec,
        field._layout,
        field._original_dims,
        field._mode_dim,
        field._nthreads,
        field._name if name is None else name,
        field._attrs if attrs is None else attrs,
        field._synthesis,
    )


def _replace_vector(
    field: SpectralVectorField,
    coefficients: xr.DataArray,
    *,
    names: tuple[Hashable | None, Hashable | None] | None = None,
    attrs: tuple[dict[str, Any], dict[str, Any]] | None = None,
) -> SpectralVectorField:
    component_names = (field._u_name, field._v_name) if names is None else names
    component_attrs = (field._u_attrs, field._v_attrs) if attrs is None else attrs
    return SpectralVectorField(
        coefficients,
        field._spec,
        field._layout,
        field._original_dims,
        field._component_dim,
        field._mode_dim,
        field._nthreads,
        component_names[0],
        component_names[1],
        dict(component_attrs[0]),
        dict(component_attrs[1]),
        field._scalar_synthesis,
        field._vector_synthesis,
    )


def _validate_selection(
    coefficient_spec: TransformSpec, selection: TransformSpec | None
) -> None:
    if not _spectral_selection_is_within(coefficient_spec, selection):
        raise ValueError("requested spectral selection exceeds the analyzed domain")


def _validate_radius(radius: float) -> None:
    if isinstance(radius, bool) or not isinstance(radius, (int, float)):
        raise TypeError("radius must be a positive finite number in metres")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be a positive finite number in metres")


def _metadata_source(name: Hashable | None, attrs: dict[str, Any]) -> xr.DataArray:
    return xr.DataArray(name=name, attrs=dict(attrs))


def _set_scalar_metadata(
    result: xr.DataArray, name: Hashable | None, attrs: dict[str, Any]
) -> xr.DataArray:
    result = result.copy(deep=False)
    result.name = name
    result.attrs = dict(attrs)
    return result


def _new_dimension(base: str, dimensions: tuple[Hashable, ...]) -> str:
    existing = {str(dimension) for dimension in dimensions}
    candidate = base
    counter = 1
    while candidate in existing:
        candidate = f"{base}{counter}"
        counter += 1
    return candidate
