# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scalar and vector regridding tests."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
import xarray as xr

import spharmgrid as sg
from spharmgrid._ducc import (
    alm_degrees,
    alm_orders,
    geometry_for,
    scalar_synthesis,
    vector_synthesis,
)
from spharmgrid._transform import TransformSpec
from spharmgrid.grids import grid_layout
from tests.conftest import scalar_field, supported_grid


def _axisymmetric_vector_field(grid: sg.Grid) -> tuple[xr.DataArray, xr.DataArray]:
    """Return degree-one and degree-two tangent-vector harmonics."""
    latitude = np.deg2rad(grid.latitude)[:, None]
    sine = np.sin(latitude)
    cosine = np.cos(latitude)
    coordinates = {"lat": grid.latitude, "lon": grid.longitude}
    shape = (1, grid.nlon)
    u = xr.DataArray(
        (5.0 * cosine + 3.0 * sine * cosine) * np.ones(shape),
        dims=("lat", "lon"),
        coords=coordinates,
        name="u",
        attrs={"standard_name": "eastward_wind", "units": "m s-1"},
    )
    v = xr.DataArray(
        (-7.0 * cosine + 2.0 * sine * cosine) * np.ones(shape),
        dims=("lat", "lon"),
        coords=coordinates,
        name="v",
        attrs={"standard_name": "northward_wind", "units": "m s-1"},
    )
    return u, v


def _scalar_mode(
    grid: sg.Grid,
    degree: int,
    order: int,
    *,
    spec: TransformSpec,
) -> xr.DataArray:
    degrees = alm_degrees(spec.lmax, spec.mmax)
    index = np.flatnonzero(
        (degrees == degree) & (alm_orders(spec.lmax, spec.mmax) == order)
    )
    assert index.size == 1
    coefficients = np.zeros((1, degrees.size), dtype=np.complex128)
    coefficients[0, index[0]] = 1.0
    layout = grid_layout(grid)
    values = scalar_synthesis(
        coefficients,
        spec=spec,
        geometry=geometry_for(grid),
        ntheta=grid.nlat,
        nphi=grid.nlon,
        phi0=layout.phi0_radians,
        nthreads=0,
    )
    return xr.DataArray(
        values,
        dims=("lat", "lon"),
        coords={"lat": grid.latitude, "lon": grid.longitude},
    )


def _vector_mode(
    grid: sg.Grid,
    degree: int,
    order: int,
    *,
    spec: TransformSpec,
) -> tuple[xr.DataArray, xr.DataArray]:
    degrees = alm_degrees(spec.lmax, spec.mmax)
    index = np.flatnonzero(
        (degrees == degree) & (alm_orders(spec.lmax, spec.mmax) == order)
    )
    assert index.size == 1
    coefficients = np.zeros((2, degrees.size), dtype=np.complex128)
    coefficients[0, index[0]] = 1.0
    layout = grid_layout(grid)
    eastward, northward = vector_synthesis(
        coefficients[0],
        coefficients[1],
        spec=spec,
        geometry=geometry_for(grid),
        ntheta=grid.nlat,
        nphi=grid.nlon,
        phi0=layout.phi0_radians,
        nthreads=0,
    )
    coordinates = {"lat": grid.latitude, "lon": grid.longitude}
    return (
        xr.DataArray(eastward, dims=("lat", "lon"), coords=coordinates),
        xr.DataArray(northward, dims=("lat", "lon"), coords=coordinates),
    )


def test_shaped_scalar_regridding_uses_bounds_and_rhomboidal_mask() -> None:
    source = sg.clenshaw_curtis_grid(8, 12, latitude_order="descending")
    target = sg.gaussian_grid(6, 12, latitude_order="descending")
    spec = TransformSpec(0, 4, 2, "trapezoidal")
    trapezoidal = _scalar_mode(source, 4, 2, spec=spec)
    outside = _scalar_mode(source, 3, 0, spec=spec)
    boundary = _scalar_mode(source, 4, 2, spec=spec)
    expected_trapezoidal = _scalar_mode(target, 4, 2, spec=spec)
    expected_boundary = _scalar_mode(target, 4, 2, spec=spec)

    trapezoidal_result = sg.regrid(trapezoidal, target, "T4x2")
    outside_result = sg.regrid(outside, target, "R2")
    boundary_result = sg.regrid(boundary, target, "R2")

    np.testing.assert_allclose(
        trapezoidal_result,
        expected_trapezoidal,
        rtol=0.0,
        atol=3.0e-13,
    )
    np.testing.assert_allclose(outside_result, 0.0, rtol=0.0, atol=3.0e-13)
    np.testing.assert_allclose(
        boundary_result,
        expected_boundary,
        rtol=0.0,
        atol=3.0e-13,
    )


def test_rhomboidal_vector_regridding_uses_the_same_domain_mask() -> None:
    source = sg.clenshaw_curtis_grid(8, 12, latitude_order="descending")
    target = sg.gaussian_grid(6, 12, latitude_order="descending")
    spec = TransformSpec(0, 4, 2, "trapezoidal")
    outside_u, outside_v = _vector_mode(source, 3, 0, spec=spec)
    boundary_u, boundary_v = _vector_mode(source, 4, 2, spec=spec)
    expected_u, expected_v = _vector_mode(target, 4, 2, spec=spec)

    outside = sg.regrid_vector(outside_u, outside_v, target, "R2")
    boundary = sg.regrid_vector(boundary_u, boundary_v, target, "R2")

    np.testing.assert_allclose(outside.u, 0.0, rtol=0.0, atol=3.0e-13)
    np.testing.assert_allclose(outside.v, 0.0, rtol=0.0, atol=3.0e-13)
    np.testing.assert_allclose(boundary.u, expected_u, rtol=0.0, atol=3.0e-13)
    np.testing.assert_allclose(boundary.v, expected_v, rtol=0.0, atol=3.0e-13)


@pytest.mark.parametrize(
    ("source_kind", "target_kind"),
    [("cc", "gl"), ("gl", "cc"), ("cc", "cc"), ("gl", "gl")],
)
def test_regridding_low_degree_field_all_supported_pairs(
    source_kind: Literal["cc", "gl"], target_kind: Literal["cc", "gl"]
) -> None:
    source = supported_grid(source_kind)
    target = supported_grid(target_kind)
    field = scalar_field(source)

    result = sg.regrid(field, target)
    expected = scalar_field(target)

    np.testing.assert_allclose(result, expected, rtol=0.0, atol=2.0e-14)
    np.testing.assert_array_equal(result.lat, target.latitude)
    np.testing.assert_array_equal(result.lon, target.longitude)


@pytest.mark.parametrize(
    ("source_kind", "target_kind"),
    [("cc", "gl"), ("gl", "cc"), ("cc", "cc"), ("gl", "gl")],
)
def test_vector_regridding_low_degree_field_all_supported_pairs(
    source_kind: Literal["cc", "gl"], target_kind: Literal["cc", "gl"]
) -> None:
    source = supported_grid(source_kind)
    target = supported_grid(target_kind)
    u, v = _axisymmetric_vector_field(source)

    result = sg.regrid_vector(u, v, target)
    expected_u, expected_v = _axisymmetric_vector_field(target)

    np.testing.assert_allclose(result.u, expected_u, rtol=0.0, atol=3.0e-14)
    np.testing.assert_allclose(result.v, expected_v, rtol=0.0, atol=3.0e-14)
    assert result.u.attrs == u.attrs
    assert result.v.attrs == v.attrs


@pytest.mark.parametrize("kind", ["cc", "gl"])
def test_same_grid_vector_filter_removes_degree_two(
    kind: Literal["cc", "gl"],
) -> None:
    grid = supported_grid(kind)
    u, v = _axisymmetric_vector_field(grid)
    latitude = np.deg2rad(grid.latitude)[:, None]
    expected_u = 5.0 * np.cos(latitude) * np.ones((1, grid.nlon))
    expected_v = -7.0 * np.cos(latitude) * np.ones((1, grid.nlon))

    result = sg.regrid_vector(u, v, grid, truncation="T1")

    np.testing.assert_allclose(result.u, expected_u, rtol=0.0, atol=3.0e-14)
    np.testing.assert_allclose(result.v, expected_v, rtol=0.0, atol=3.0e-14)
    np.testing.assert_array_equal(result.lat, grid.latitude)
    np.testing.assert_array_equal(result.lon, grid.longitude)


@pytest.mark.parametrize("kind", ["cc", "gl"])
def test_same_grid_vector_taper_has_expected_upper_degree_response(
    kind: Literal["cc", "gl"],
) -> None:
    grid = supported_grid(kind)
    mixed_u, mixed_v = _axisymmetric_vector_field(grid)
    latitude = np.deg2rad(grid.latitude)[:, None]
    shape = (1, grid.nlon)
    u = mixed_u.copy(data=3.0 * np.sin(latitude) * np.cos(latitude) * np.ones(shape))
    v = mixed_v.copy(data=2.0 * np.sin(latitude) * np.cos(latitude) * np.ones(shape))

    hard = sg.regrid_vector(u, v, grid, truncation="T2-2")
    tapered = sg.regrid_vector(u, v, grid, truncation="T2", taper=0.1)

    np.testing.assert_allclose(hard.u, u, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(hard.v, v, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(tapered.u, 0.1 * u, rtol=0.0, atol=1.0e-15)
    np.testing.assert_allclose(tapered.v, 0.1 * v, rtol=0.0, atol=1.0e-15)


def test_vector_regridding_selection_and_taper() -> None:
    source = supported_grid("cc", latitude_order="descending", lon0=-180.0)
    target = supported_grid("gl", latitude_order="descending", lon0=-180.0)
    u, v = _axisymmetric_vector_field(source)
    latitude = np.deg2rad(source.latitude)[:, None]
    degree_two_u = 3.0 * np.sin(latitude) * np.cos(latitude)
    degree_two_v = 2.0 * np.sin(latitude) * np.cos(latitude)
    u = u.copy(data=degree_two_u * np.ones((1, source.nlon)))
    v = v.copy(data=degree_two_v * np.ones((1, source.nlon)))

    direct = sg.regrid_vector(u, v, target, truncation="T2", taper=0.1)
    target_latitude = np.deg2rad(target.latitude)[:, None]
    expected_u = (
        0.1
        * 3.0
        * np.sin(target_latitude)
        * np.cos(target_latitude)
        * np.ones((1, target.nlon))
    )
    expected_v = (
        0.1
        * 2.0
        * np.sin(target_latitude)
        * np.cos(target_latitude)
        * np.ones((1, target.nlon))
    )

    np.testing.assert_allclose(direct.u, expected_u, rtol=0.0, atol=5.0e-16)
    np.testing.assert_allclose(direct.v, expected_v, rtol=0.0, atol=5.0e-16)
    np.testing.assert_array_equal(direct.lat, target.latitude)
    np.testing.assert_array_equal(direct.lon, target.longitude)


def test_vector_regridding_round_trip_is_band_limited() -> None:
    source = supported_grid("cc")
    intermediate = supported_grid("gl")
    u, v = _axisymmetric_vector_field(source)

    regridded = sg.regrid_vector(u, v, intermediate, truncation="T2")
    restored = sg.regrid_vector(
        regridded.u,
        regridded.v,
        source,
        truncation="T2",
    )

    np.testing.assert_allclose(restored.u, u, rtol=0.0, atol=3.0e-14)
    np.testing.assert_allclose(restored.v, v, rtol=0.0, atol=3.0e-14)


def test_combined_filter_and_regrid_matches_sequential_result() -> None:
    source = scalar_field(supported_grid("cc"))
    target = supported_grid("gl")

    result = sg.regrid(source, target, truncation="T2", taper=1.0)
    expected = sg.regrid(sg.filter(source, "T2"), target, truncation="T2")

    np.testing.assert_allclose(result, expected, rtol=0.0, atol=5.0e-15)


def test_regridding_restores_descending_and_shifted_target_coordinates() -> None:
    source_grid = supported_grid("cc", latitude_order="descending", lon0=-180.0)
    target_grid = sg.gaussian_grid(
        16,
        36,
        latitude_order="descending",
        lon0=-180.0,
    )
    source = scalar_field(source_grid)

    result = sg.regrid(source, target_grid, truncation="T3")
    expected = scalar_field(target_grid)

    np.testing.assert_allclose(result, expected, rtol=0.0, atol=1.0e-14)
    np.testing.assert_array_equal(result.lat, expected.lat)
    np.testing.assert_array_equal(result.lon, expected.lon)
    assert result.lat.attrs["standard_name"] == "latitude"
    assert result.lon.attrs["standard_name"] == "longitude"
