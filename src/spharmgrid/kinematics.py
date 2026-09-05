"""Wind kinematics, potentials, and inverse vector transforms using DUCC0."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, cast

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ._ducc import (
    TransformSpec,
    alm_degrees,
    geometry_for,
    scalar_analysis,
    scalar_synthesis,
    vector_analysis,
    vector_synthesis,
)
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
from .metadata import (
    ScalarSource,
    identify_scalar_source,
    wind_component_metadata,
    with_output_metadata,
)
from .operators import EARTH_RADIUS_M
from .spectral import transform_spec

WindSource = Literal["vorticity_divergence", "potentials"]


def vorticity(
    u: xr.DataArray,
    v: xr.DataArray,
    *,
    output: str = "vo",
    radius: float = EARTH_RADIUS_M,
    nthreads: int | None = None,
) -> xr.DataArray:
    """Compute relative vorticity from eastward and northward wind."""
    vo, _ = _kinematic_fields(u, v, radius=radius, nthreads=nthreads)
    return with_output_metadata(vo, "vo", output)


def divergence(
    u: xr.DataArray,
    v: xr.DataArray,
    *,
    output: str = "d",
    radius: float = EARTH_RADIUS_M,
    nthreads: int | None = None,
) -> xr.DataArray:
    """Compute horizontal wind divergence from eastward and northward wind."""
    _, div = _kinematic_fields(u, v, radius=radius, nthreads=nthreads)
    return with_output_metadata(div, "d", output)


def kinematics(
    u: xr.DataArray,
    v: xr.DataArray,
    *,
    vorticity: str = "vo",
    divergence: str = "d",
    radius: float = EARTH_RADIUS_M,
    nthreads: int | None = None,
) -> xr.Dataset:
    """Compute relative vorticity and divergence from one spin-1 analysis."""
    _validate_distinct_names(vorticity, divergence)
    vo, div = _kinematic_fields(u, v, radius=radius, nthreads=nthreads)
    vo = with_output_metadata(vo, "vo", vorticity)
    div = with_output_metadata(div, "d", divergence)
    return xr.Dataset({vorticity: vo, divergence: div})


def streamfunction(
    u: xr.DataArray,
    v: xr.DataArray,
    *,
    output: str = "strf",
    radius: float = EARTH_RADIUS_M,
    nthreads: int | None = None,
) -> xr.DataArray:
    """Compute the horizontal streamfunction from a wind field."""
    psi, _ = _potential_fields(u, v, radius=radius, nthreads=nthreads)
    return with_output_metadata(psi, "strf", output)


def velocity_potential(
    u: xr.DataArray,
    v: xr.DataArray,
    *,
    output: str = "vp",
    radius: float = EARTH_RADIUS_M,
    nthreads: int | None = None,
) -> xr.DataArray:
    """Compute the horizontal velocity potential from a wind field."""
    _, chi = _potential_fields(u, v, radius=radius, nthreads=nthreads)
    return with_output_metadata(chi, "vp", output)


def potentials(
    u: xr.DataArray,
    v: xr.DataArray,
    *,
    streamfunction: str = "strf",
    velocity_potential: str = "vp",
    radius: float = EARTH_RADIUS_M,
    nthreads: int | None = None,
) -> xr.Dataset:
    """Compute streamfunction and velocity potential from one wind analysis.

    For positive total degree, the returned fields satisfy
    ``vorticity = laplacian(streamfunction)`` and
    ``divergence = laplacian(velocity_potential)``.  Their additive constants
    are set by zeroing degree zero.
    """
    _validate_distinct_names(streamfunction, velocity_potential)
    psi, chi = _potential_fields(u, v, radius=radius, nthreads=nthreads)
    psi = with_output_metadata(psi, "strf", streamfunction)
    chi = with_output_metadata(chi, "vp", velocity_potential)
    return xr.Dataset({streamfunction: psi, velocity_potential: chi})


def rotational_wind(
    field: xr.DataArray,
    *,
    quantity: Literal["vorticity", "streamfunction"] | None = None,
    eastward: str = "u_rotational",
    northward: str = "v_rotational",
    radius: float = EARTH_RADIUS_M,
    nthreads: int | None = None,
) -> xr.Dataset:
    """Recover rotational wind from relative vorticity or streamfunction."""
    field = require_dataarray(field)
    source = identify_scalar_source(
        field,
        allowed=("vorticity", "streamfunction"),
        quantity=quantity,
    )
    _validate_distinct_names(eastward, northward)
    u, v = _single_source_wind(
        field,
        source=source,
        kind="rotational",
        radius=radius,
        nthreads=nthreads,
    )
    return _wind_dataset(u, v, eastward, northward, "rotational")


def divergent_wind(
    field: xr.DataArray,
    *,
    quantity: Literal["divergence", "velocity_potential"] | None = None,
    eastward: str = "u_divergent",
    northward: str = "v_divergent",
    radius: float = EARTH_RADIUS_M,
    nthreads: int | None = None,
) -> xr.Dataset:
    """Recover divergent wind from divergence or velocity potential."""
    field = require_dataarray(field)
    source = identify_scalar_source(
        field,
        allowed=("divergence", "velocity_potential"),
        quantity=quantity,
    )
    _validate_distinct_names(eastward, northward)
    u, v = _single_source_wind(
        field,
        source=source,
        kind="divergent",
        radius=radius,
        nthreads=nthreads,
    )
    return _wind_dataset(u, v, eastward, northward, "divergent")


def wind(
    first: xr.DataArray,
    second: xr.DataArray,
    *,
    source: WindSource | None = None,
    eastward: str = "u",
    northward: str = "v",
    radius: float = EARTH_RADIUS_M,
    nthreads: int | None = None,
) -> xr.Dataset:
    """Reconstruct wind from ``vo``/``d`` or ``strf``/``vp`` scalar fields.

    When ``source`` is omitted, exact CF metadata or canonical short names
    must identify one complete source representation.  Explicit ``source``
    interprets ``first, second`` in the documented order.
    """
    first = require_dataarray(first, "first")
    second = require_dataarray(second, "second")
    _validate_distinct_names(eastward, northward)
    resolved_source, scalar_one, scalar_two = _resolve_wind_inputs(
        first, second, source
    )
    u, v = _two_source_wind(
        scalar_one,
        scalar_two,
        source=resolved_source,
        radius=radius,
        nthreads=nthreads,
    )
    u = with_output_metadata(u, "u", eastward)
    v = with_output_metadata(v, "v", northward)
    return xr.Dataset({eastward: u, northward: v})


def _kinematic_fields(
    u: xr.DataArray,
    v: xr.DataArray,
    *,
    radius: float,
    nthreads: int | None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Compute vorticity then divergence while sharing vector analysis."""
    _validate_radius(radius)
    layout, canonical_u, canonical_v = _wind_inputs(u, v)
    spec = _vector_spec(layout)
    degrees = alm_degrees(spec.lmax, spec.mmax).astype(np.float64)
    scale = np.sqrt(degrees * (degrees + 1.0)) / radius

    def transform(
        frame_u: NDArray[np.generic], frame_v: NDArray[np.generic], threads: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        vector_alm = vector_analysis(
            frame_u,
            frame_v,
            spec=spec,
            geometry=geometry_for(layout.grid),
            phi0=layout.transform_layout.phi0_radians,
            nthreads=threads,
        )
        divergence_alm = -scale * vector_alm[0]
        vorticity_alm = -scale * vector_alm[1]
        vorticity_map = scalar_synthesis(
            vorticity_alm[np.newaxis, :],
            spec=spec,
            geometry=geometry_for(layout.grid),
            ntheta=layout.grid.nlat,
            nphi=layout.grid.nlon,
            phi0=layout.transform_layout.phi0_radians,
            nthreads=threads,
        )
        divergence_map = scalar_synthesis(
            divergence_alm[np.newaxis, :],
            spec=spec,
            geometry=geometry_for(layout.grid),
            ntheta=layout.grid.nlat,
            nphi=layout.grid.nlon,
            phi0=layout.transform_layout.phi0_radians,
            nthreads=threads,
        )
        return vorticity_map, divergence_map

    return _two_wind_to_scalar_outputs(
        u,
        layout,
        canonical_u,
        canonical_v,
        transform,
        nthreads=nthreads,
    )


def _potential_fields(
    u: xr.DataArray,
    v: xr.DataArray,
    *,
    radius: float,
    nthreads: int | None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Compute streamfunction then velocity potential from one vector analysis."""
    _validate_radius(radius)
    layout, canonical_u, canonical_v = _wind_inputs(u, v)
    spec = _vector_spec(layout)
    degrees = alm_degrees(spec.lmax, spec.mmax).astype(np.float64)
    scale = np.sqrt(degrees * (degrees + 1.0)) / radius
    inverse_laplacian = _inverse_laplacian_multiplier(degrees, radius)

    def transform(
        frame_u: NDArray[np.generic], frame_v: NDArray[np.generic], threads: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        vector_alm = vector_analysis(
            frame_u,
            frame_v,
            spec=spec,
            geometry=geometry_for(layout.grid),
            phi0=layout.transform_layout.phi0_radians,
            nthreads=threads,
        )
        divergence_alm = -scale * vector_alm[0]
        vorticity_alm = -scale * vector_alm[1]
        streamfunction_alm = inverse_laplacian * vorticity_alm
        velocity_potential_alm = inverse_laplacian * divergence_alm
        streamfunction_map = scalar_synthesis(
            streamfunction_alm[np.newaxis, :],
            spec=spec,
            geometry=geometry_for(layout.grid),
            ntheta=layout.grid.nlat,
            nphi=layout.grid.nlon,
            phi0=layout.transform_layout.phi0_radians,
            nthreads=threads,
        )
        velocity_potential_map = scalar_synthesis(
            velocity_potential_alm[np.newaxis, :],
            spec=spec,
            geometry=geometry_for(layout.grid),
            ntheta=layout.grid.nlat,
            nphi=layout.grid.nlon,
            phi0=layout.transform_layout.phi0_radians,
            nthreads=threads,
        )
        return streamfunction_map, velocity_potential_map

    return _two_wind_to_scalar_outputs(
        u,
        layout,
        canonical_u,
        canonical_v,
        transform,
        nthreads=nthreads,
    )


def _single_source_wind(
    field: xr.DataArray,
    *,
    source: ScalarSource,
    kind: Literal["rotational", "divergent"],
    radius: float,
    nthreads: int | None,
) -> tuple[xr.DataArray, xr.DataArray]:
    _validate_radius(radius)
    layout = field_layout(field)
    spec = _vector_spec(layout)
    degrees = alm_degrees(spec.lmax, spec.mmax).astype(np.float64)
    scale = np.sqrt(degrees * (degrees + 1.0)) / radius
    laplacian = -(degrees * (degrees + 1.0)) / radius**2

    def transform(
        frame: NDArray[np.generic], threads: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        scalar_alm = scalar_analysis(
            frame,
            spec=spec,
            geometry=geometry_for(layout.grid),
            phi0=layout.transform_layout.phi0_radians,
            nthreads=threads,
        )[0]
        e_alm = np.zeros_like(scalar_alm)
        b_alm = np.zeros_like(scalar_alm)
        nonzero = scale > 0.0
        if kind == "rotational":
            vorticity_alm = (
                scalar_alm if source == "vorticity" else laplacian * scalar_alm
            )
            b_alm[nonzero] = -vorticity_alm[nonzero] / scale[nonzero]
        else:
            divergence_alm = (
                scalar_alm if source == "divergence" else laplacian * scalar_alm
            )
            e_alm[nonzero] = -divergence_alm[nonzero] / scale[nonzero]
        return vector_synthesis(
            e_alm,
            b_alm,
            spec=spec,
            geometry=geometry_for(layout.grid),
            ntheta=layout.grid.nlat,
            nphi=layout.grid.nlon,
            phi0=layout.transform_layout.phi0_radians,
            nthreads=threads,
        )

    return _single_scalar_to_wind(field, layout, transform, nthreads=nthreads)


def _two_source_wind(
    first: xr.DataArray,
    second: xr.DataArray,
    *,
    source: WindSource,
    radius: float,
    nthreads: int | None,
) -> tuple[xr.DataArray, xr.DataArray]:
    _validate_radius(radius)
    layout, canonical_first, canonical_second = _paired_scalar_inputs(first, second)
    spec = _vector_spec(layout)
    degrees = alm_degrees(spec.lmax, spec.mmax).astype(np.float64)
    scale = np.sqrt(degrees * (degrees + 1.0)) / radius
    laplacian = -(degrees * (degrees + 1.0)) / radius**2

    def transform(
        frame_first: NDArray[np.generic],
        frame_second: NDArray[np.generic],
        threads: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        first_alm = scalar_analysis(
            frame_first,
            spec=spec,
            geometry=geometry_for(layout.grid),
            phi0=layout.transform_layout.phi0_radians,
            nthreads=threads,
        )[0]
        second_alm = scalar_analysis(
            frame_second,
            spec=spec,
            geometry=geometry_for(layout.grid),
            phi0=layout.transform_layout.phi0_radians,
            nthreads=threads,
        )[0]
        if source == "vorticity_divergence":
            vorticity_alm, divergence_alm = first_alm, second_alm
        else:
            vorticity_alm = laplacian * first_alm
            divergence_alm = laplacian * second_alm
        e_alm = np.zeros_like(divergence_alm)
        b_alm = np.zeros_like(vorticity_alm)
        nonzero = scale > 0.0
        e_alm[nonzero] = -divergence_alm[nonzero] / scale[nonzero]
        b_alm[nonzero] = -vorticity_alm[nonzero] / scale[nonzero]
        return vector_synthesis(
            e_alm,
            b_alm,
            spec=spec,
            geometry=geometry_for(layout.grid),
            ntheta=layout.grid.nlat,
            nphi=layout.grid.nlon,
            phi0=layout.transform_layout.phi0_radians,
            nthreads=threads,
        )

    return _two_scalar_to_wind(
        first,
        layout,
        canonical_first,
        canonical_second,
        transform,
        nthreads=nthreads,
    )


def _wind_inputs(
    u: xr.DataArray, v: xr.DataArray
) -> tuple[FieldLayout, xr.DataArray, xr.DataArray]:
    u = require_dataarray(u, "u")
    v = require_dataarray(v, "v")
    layout = field_layout(u)
    v_layout = field_layout(v)
    _require_matching_layouts(layout, v_layout, "u", "v")
    canonical_u = layout.canonicalize(u)
    canonical_v = v_layout.canonicalize(v)
    return layout, *exact_align(canonical_u, canonical_v, names=("u", "v"))


def _paired_scalar_inputs(
    first: xr.DataArray, second: xr.DataArray
) -> tuple[FieldLayout, xr.DataArray, xr.DataArray]:
    layout = field_layout(first)
    second_layout = field_layout(second)
    _require_matching_layouts(layout, second_layout, "first", "second")
    canonical_first = layout.canonicalize(first)
    canonical_second = second_layout.canonicalize(second)
    return layout, *exact_align(
        canonical_first,
        canonical_second,
        names=("first", "second"),
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


def _vector_spec(layout: FieldLayout) -> TransformSpec:
    spec = transform_spec(layout.grid, layout.grid, None)
    if spec.lmax < 1:
        raise ValueError("wind transforms require a grid supporting total degree l=1")
    return spec


def _two_wind_to_scalar_outputs(
    original: xr.DataArray,
    layout: FieldLayout,
    canonical_u: xr.DataArray,
    canonical_v: xr.DataArray,
    transform: Callable[
        [NDArray[np.generic], NDArray[np.generic], int],
        tuple[NDArray[np.float64], NDArray[np.float64]],
    ],
    *,
    nthreads: int | None,
) -> tuple[xr.DataArray, xr.DataArray]:
    dask_field = canonical_u if canonical_u.chunks is not None else canonical_v
    threads = resolve_nthreads(nthreads, dask_backed=dask_field.chunks is not None)

    def kernel(
        frame_u: NDArray[np.generic], frame_v: NDArray[np.generic]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return transform(frame_u, frame_v, threads)

    outputs = xr.apply_ufunc(
        kernel,
        canonical_u,
        canonical_v,
        input_core_dims=[
            [layout.latitude_dim, layout.longitude_dim],
            [layout.latitude_dim, layout.longitude_dim],
        ],
        output_core_dims=[
            [layout.latitude_dim, layout.longitude_dim],
            [layout.latitude_dim, layout.longitude_dim],
        ],
        vectorize=True,
        output_dtypes=[np.float64, np.float64],
        **apply_ufunc_options(dask_field),
    )
    first, second = cast(tuple[xr.DataArray, xr.DataArray], outputs)
    return (
        restore_output(
            first, source=layout, target=layout, original_dims=original.dims
        ),
        restore_output(
            second, source=layout, target=layout, original_dims=original.dims
        ),
    )


def _single_scalar_to_wind(
    field: xr.DataArray,
    layout: FieldLayout,
    transform: Callable[
        [NDArray[np.generic], int], tuple[NDArray[np.float64], NDArray[np.float64]]
    ],
    *,
    nthreads: int | None,
) -> tuple[xr.DataArray, xr.DataArray]:
    canonical = layout.canonicalize(field)
    threads = resolve_nthreads(nthreads, dask_backed=canonical.chunks is not None)

    def kernel(
        frame: NDArray[np.generic],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return transform(frame, threads)

    outputs = xr.apply_ufunc(
        kernel,
        canonical,
        input_core_dims=[[layout.latitude_dim, layout.longitude_dim]],
        output_core_dims=[
            [layout.latitude_dim, layout.longitude_dim],
            [layout.latitude_dim, layout.longitude_dim],
        ],
        vectorize=True,
        output_dtypes=[np.float64, np.float64],
        **apply_ufunc_options(canonical),
    )
    u, v = cast(tuple[xr.DataArray, xr.DataArray], outputs)
    return (
        restore_output(u, source=layout, target=layout, original_dims=field.dims),
        restore_output(v, source=layout, target=layout, original_dims=field.dims),
    )


def _two_scalar_to_wind(
    original: xr.DataArray,
    layout: FieldLayout,
    canonical_first: xr.DataArray,
    canonical_second: xr.DataArray,
    transform: Callable[
        [NDArray[np.generic], NDArray[np.generic], int],
        tuple[NDArray[np.float64], NDArray[np.float64]],
    ],
    *,
    nthreads: int | None,
) -> tuple[xr.DataArray, xr.DataArray]:
    dask_field = (
        canonical_first if canonical_first.chunks is not None else canonical_second
    )
    threads = resolve_nthreads(nthreads, dask_backed=dask_field.chunks is not None)

    def kernel(
        first: NDArray[np.generic], second: NDArray[np.generic]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return transform(first, second, threads)

    outputs = xr.apply_ufunc(
        kernel,
        canonical_first,
        canonical_second,
        input_core_dims=[
            [layout.latitude_dim, layout.longitude_dim],
            [layout.latitude_dim, layout.longitude_dim],
        ],
        output_core_dims=[
            [layout.latitude_dim, layout.longitude_dim],
            [layout.latitude_dim, layout.longitude_dim],
        ],
        vectorize=True,
        output_dtypes=[np.float64, np.float64],
        **apply_ufunc_options(dask_field),
    )
    u, v = cast(tuple[xr.DataArray, xr.DataArray], outputs)
    return (
        restore_output(u, source=layout, target=layout, original_dims=original.dims),
        restore_output(v, source=layout, target=layout, original_dims=original.dims),
    )


def _resolve_wind_inputs(
    first: xr.DataArray, second: xr.DataArray, source: WindSource | None
) -> tuple[WindSource, xr.DataArray, xr.DataArray]:
    if source is not None:
        if source not in {"vorticity_divergence", "potentials"}:
            raise ValueError("source must be 'vorticity_divergence' or 'potentials'")
        return source, first, second

    first_source = _identify_any_scalar_source(first)
    second_source = _identify_any_scalar_source(second)
    pair = {first_source, second_source}
    if pair == {"vorticity", "divergence"}:
        return (
            "vorticity_divergence",
            first if first_source == "vorticity" else second,
            second if second_source == "divergence" else first,
        )
    if pair == {"streamfunction", "velocity_potential"}:
        return (
            "potentials",
            first if first_source == "streamfunction" else second,
            second if second_source == "velocity_potential" else first,
        )
    raise ValueError(
        "could not infer a complete wind source; use source='vorticity_divergence' "
        "or source='potentials'"
    )


def _identify_any_scalar_source(field: xr.DataArray) -> ScalarSource:
    return identify_scalar_source(
        field,
        allowed=(
            "vorticity",
            "divergence",
            "streamfunction",
            "velocity_potential",
        ),
    )


def _inverse_laplacian_multiplier(
    degrees: NDArray[np.float64], radius: float
) -> NDArray[np.float64]:
    multiplier = np.zeros_like(degrees)
    nonzero = degrees > 0.0
    multiplier[nonzero] = -(radius**2) / (degrees[nonzero] * (degrees[nonzero] + 1.0))
    return multiplier


def _wind_dataset(
    u: xr.DataArray,
    v: xr.DataArray,
    eastward: str,
    northward: str,
    kind: Literal["rotational", "divergent"],
) -> xr.Dataset:
    u = u.copy(deep=False)
    v = v.copy(deep=False)
    u.name = eastward
    v.name = northward
    u.attrs = wind_component_metadata("eastward", kind)
    v.attrs = wind_component_metadata("northward", kind)
    return xr.Dataset({eastward: u, northward: v})


def _validate_radius(radius: float) -> None:
    if isinstance(radius, bool) or not isinstance(radius, (int, float)):
        raise TypeError("radius must be a positive finite number in metres")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be a positive finite number in metres")


def _validate_distinct_names(first: str, second: str) -> None:
    if not first or not second:
        raise ValueError("output names must be non-empty")
    if first == second:
        raise ValueError("two output names must be distinct")
