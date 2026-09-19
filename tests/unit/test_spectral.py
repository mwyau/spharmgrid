# SPDX-FileCopyrightText: Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Analytic scalar filtering, regridding, and operator tests."""

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
from spharmgrid.spectral import apply_spectral_selection, resolve_transform_spec
from tests.conftest import degree_one_field, scalar_field, supported_grid


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


@pytest.mark.parametrize(
    ("notation", "expected"),
    [
        ("T42", TransformSpec(0, 42, 42, "triangular")),
        ("t6-42", TransformSpec(6, 42, 42, "triangular")),
        ("T6–42", TransformSpec(6, 42, 42, "triangular")),
    ],
)
def test_parse_spectral(notation: str, expected: TransformSpec) -> None:
    result = sg.parse_spectral(notation)

    assert result == expected


@pytest.mark.parametrize(
    ("notation", "expected"),
    [
        ("T42x10", TransformSpec(0, 42, 10, "trapezoidal")),
        ("t42X10", TransformSpec(0, 42, 10, "trapezoidal")),
        ("R42", TransformSpec(0, 84, 42, "rhomboidal")),
        ("r42", TransformSpec(0, 84, 42, "rhomboidal")),
    ],
)
def test_parse_nontriangular_spectral(
    notation: str,
    expected: TransformSpec,
) -> None:
    result = sg.parse_spectral(notation)

    assert result == expected


@pytest.mark.parametrize(
    "notation",
    ["", "T", "T42x", "T42-10", "T42x43", "R42x10", "Q42", "T5-42x10"],
)
def test_parse_spectral_rejects_malformed_shapes(notation: str) -> None:
    with pytest.raises(
        ValueError,
        match="spectral notation|spectral bounds|transform limits",
    ):
        sg.parse_spectral(notation)


def test_transform_spec_resolves_supported_shapes_and_full_domain() -> None:
    grid = sg.gaussian_grid(10, 20)

    assert resolve_transform_spec(grid, grid, sg.parse_spectral("T8")) == (
        TransformSpec(0, 8, 8, "triangular")
    )
    assert resolve_transform_spec(grid, grid, sg.parse_spectral("T8x3")) == (
        TransformSpec(0, 8, 3, "trapezoidal")
    )
    assert resolve_transform_spec(grid, grid, sg.parse_spectral("R4")) == (
        TransformSpec(0, 8, 4, "rhomboidal")
    )

    latitude_limited_by_grid = sg.gaussian_grid(8, 6)
    assert resolve_transform_spec(
        latitude_limited_by_grid, latitude_limited_by_grid, None
    ) == (TransformSpec(0, 7, 2, "trapezoidal"))


def test_nontriangular_capabilities_check_latitude_and_longitude_independently() -> (
    None
):
    low_latitude = sg.gaussian_grid(5, 40)
    high_bandwidth = sg.gaussian_grid(8, 40)
    low_longitude = sg.gaussian_grid(8, 6)

    with pytest.raises(ValueError, match="source latitude"):
        resolve_transform_spec(low_latitude, high_bandwidth, sg.parse_spectral("T5x2"))
    with pytest.raises(ValueError, match="target longitude"):
        resolve_transform_spec(high_bandwidth, low_longitude, sg.parse_spectral("T5x3"))


def test_alm_indices_follow_ducc_packed_order_and_rhomboidal_mask() -> None:
    degrees = alm_degrees(4, 2)
    orders = alm_orders(4, 2)
    assert list(zip(degrees.tolist(), orders.tolist(), strict=True)) == [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 1),
        (2, 2),
        (3, 2),
        (4, 2),
    ]

    spec = sg.parse_spectral("R2")
    coefficients = np.ones((2, degrees.size), dtype=np.complex128)
    selected = apply_spectral_selection(coefficients, spec, taper=None)
    expected = (degrees - orders <= spec.lmax - spec.mmax).astype(np.float64)
    np.testing.assert_allclose(selected, np.tile(expected, (2, 1)))

    tapered = apply_spectral_selection(coefficients, spec, taper=0.1)
    boundary = (degrees == 4) & (orders == 2)
    np.testing.assert_allclose(tapered[:, boundary], 0.1)
    np.testing.assert_allclose(tapered[:, (degrees == 3) & (orders == 0)], 0.0)


def test_trapezoidal_filter_retains_the_lmax_corner() -> None:
    grid = sg.clenshaw_curtis_grid(8, 12, latitude_order="descending")
    spec = TransformSpec(0, 4, 2, "trapezoidal")
    field = _scalar_mode(grid, 4, 2, spec=spec)

    result = sg.filter(field, "T4x2")

    np.testing.assert_allclose(result, field, rtol=0.0, atol=2.0e-13)


def test_rhomboidal_filter_removes_outside_modes_and_keeps_boundary_modes() -> None:
    grid = sg.clenshaw_curtis_grid(8, 12, latitude_order="descending")
    spec = TransformSpec(0, 4, 2, "trapezoidal")
    outside = _scalar_mode(grid, 3, 0, spec=spec)
    boundary = _scalar_mode(grid, 4, 2, spec=spec)

    outside_result = sg.filter(outside, "R2")
    boundary_result = sg.filter(boundary, "R2")
    tapered_boundary = sg.filter(boundary, "R2", taper=0.1)

    np.testing.assert_allclose(outside_result, 0.0, rtol=0.0, atol=2.0e-13)
    np.testing.assert_allclose(boundary_result, boundary, rtol=0.0, atol=2.0e-13)
    np.testing.assert_allclose(tapered_boundary, 0.1 * boundary, rtol=0.0, atol=2.0e-13)


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


def test_t5_42_remains_a_triangular_degree_band() -> None:
    grid = sg.gaussian_grid(44, 86, latitude_order="descending")
    spec = TransformSpec(0, 42, 42, "triangular")
    retained = _scalar_mode(grid, 6, 0, spec=spec)
    removed = _scalar_mode(grid, 3, 0, spec=spec)
    field = retained + removed

    parsed = sg.filter(field, "T5-42")
    explicit = sg.filter(field, lmin=5, lmax=42)

    np.testing.assert_allclose(parsed, retained, rtol=0.0, atol=3.0e-12)
    np.testing.assert_allclose(explicit, parsed, rtol=0.0, atol=3.0e-12)


@pytest.mark.parametrize("kind", ["cc", "gl"])
def test_hard_degree_selection_and_bandpass(kind: Literal["cc", "gl"]) -> None:
    grid = supported_grid(kind)
    latitude = np.sin(np.deg2rad(grid.latitude))[:, None]
    degree_zero = np.ones((grid.nlat, grid.nlon))
    degree_one = latitude * np.ones((1, grid.nlon))
    degree_two = (3.0 * latitude**2 - 1.0) / 2.0 * np.ones((1, grid.nlon))
    field = xr.DataArray(
        degree_zero + 2.0 * degree_one + 3.0 * degree_two,
        dims=("lat", "lon"),
        coords={"lat": grid.latitude, "lon": grid.longitude},
    )

    low = sg.filter(field, "T1")
    band = sg.filter(field, "T2-2")
    explicit_band = sg.filter(field, lmin=2, lmax=2)

    np.testing.assert_allclose(
        low, degree_zero + 2.0 * degree_one, rtol=0.0, atol=5.0e-15
    )
    np.testing.assert_allclose(band, 3.0 * degree_two, rtol=0.0, atol=1.0e-14)
    xr.testing.assert_identical(explicit_band, band)


@pytest.mark.parametrize("kind", ["cc", "gl"])
def test_t42_is_accepted_on_a_grid_that_represents_it(
    kind: Literal["cc", "gl"],
) -> None:
    grid = sg.clenshaw_curtis_grid(44, 85) if kind == "cc" else sg.gaussian_grid(43, 85)
    field = scalar_field(grid)

    result = sg.filter(field, "T42")

    np.testing.assert_allclose(result, field, rtol=0.0, atol=2.0e-13)


@pytest.mark.parametrize("kind", ["cc", "gl"])
def test_taper_response_at_upper_retained_degree(kind: Literal["cc", "gl"]) -> None:
    grid = supported_grid(kind)
    latitude = np.sin(np.deg2rad(grid.latitude))[:, None]
    degree_two = (3.0 * latitude**2 - 1.0) / 2.0 * np.ones((1, grid.nlon))
    field = xr.DataArray(
        degree_two,
        dims=("lat", "lon"),
        coords={"lat": grid.latitude, "lon": grid.longitude},
    )

    tapered = sg.filter(field, "T2", taper=0.1)

    np.testing.assert_allclose(tapered, 0.1 * field, rtol=0.0, atol=5.0e-16)


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


@pytest.mark.parametrize("kind", ["cc", "gl"])
def test_gradient_laplacian_and_inverse_laplacian_of_degree_one(
    kind: Literal["cc", "gl"],
) -> None:
    grid = supported_grid(kind)
    field = degree_one_field(grid)
    latitude = np.deg2rad(grid.latitude)[:, None]
    radius = sg.EARTH_RADIUS_M

    gradient = sg.gradient(field)
    laplacian = sg.laplacian(field)
    inverse = sg.inverse_laplacian(field)

    expected_north = np.cos(latitude) / radius * np.ones((1, grid.nlon))
    np.testing.assert_allclose(gradient.gradient_eastward, 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        gradient.gradient_northward, expected_north, rtol=0.0, atol=1.0e-20
    )
    np.testing.assert_allclose(
        laplacian, -2.0 * field / radius**2, rtol=0.0, atol=5.0e-26
    )
    # The inverse multiplier is O(R**2), so coefficient roundoff is amplified
    # to a few 1e-3 in physical space across the supported platforms.
    np.testing.assert_allclose(
        inverse,
        -(radius**2) * field / 2.0,
        rtol=0.0,
        atol=1.0e-2,
    )


@pytest.mark.parametrize("kind", ["cc", "gl"])
def test_inverse_laplacian_sets_degree_zero_to_zero(kind: Literal["cc", "gl"]) -> None:
    grid = supported_grid(kind)
    field = degree_one_field(grid) + 7.0

    inverse = sg.inverse_laplacian(field)
    restored = sg.laplacian(inverse)

    np.testing.assert_allclose(
        inverse,
        -(sg.EARTH_RADIUS_M**2) * degree_one_field(grid) / 2.0,
        rtol=0.0,
        # Roundoff in the degree-one coefficient is amplified by R**2 after
        # analyzing a field that also contains a large constant.
        atol=1.0e-1,
    )
    np.testing.assert_allclose(restored, degree_one_field(grid), rtol=0.0, atol=1.0e-12)


def test_scalar_laplacian_units_simplify_operator_chains() -> None:
    field = degree_one_field(supported_grid("cc"))
    field.attrs["units"] = "K"

    laplacian = sg.laplacian(field)
    restored = sg.inverse_laplacian(laplacian)
    inverse = sg.inverse_laplacian(field)
    inverse_restored = sg.laplacian(inverse)

    assert laplacian.attrs["units"] == "K m-2"
    assert restored.attrs["units"] == "K"
    assert inverse.attrs["units"] == "K m2"
    assert inverse_restored.attrs["units"] == "K"


def test_spectral_error_cases_are_explicit() -> None:
    field = scalar_field(supported_grid("cc"))

    with pytest.raises(ValueError, match="either truncation"):
        sg.filter(field, "T2", lmin=0, lmax=2)
    with pytest.raises(ValueError, match="both lmin"):
        sg.filter(field, lmax=2)
    with pytest.raises(ValueError, match="taper"):
        sg.filter(field, "T2", taper=0.0)
    with pytest.raises(ValueError, match="exceeds"):
        sg.filter(field, "T99")
    with pytest.raises(ValueError, match="exceeds"):
        sg.regrid(field, sg.clenshaw_curtis_grid(9, 18), truncation="T8")
    with pytest.raises(ValueError, match="must be distinct"):
        sg.gradient(field, eastward="gradient", northward="gradient")
