"""Analytic scalar filtering, regridding, and operator tests."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
import xarray as xr

import spharmgrid as sg
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


@pytest.mark.parametrize(
    ("notation", "expected"),
    [("T42", (0, 42)), ("t6-42", (6, 42)), ("T6–42", (6, 42))],
)
def test_parse_spectral(notation: str, expected: tuple[int, int]) -> None:
    result = sg.parse_spectral(notation)

    assert (result.lmin, result.lmax) == expected


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

    np.testing.assert_allclose(low, degree_zero + 2.0 * degree_one, atol=2.0e-11)
    np.testing.assert_allclose(band, 3.0 * degree_two, atol=2.0e-11)
    xr.testing.assert_identical(explicit_band, band)


@pytest.mark.parametrize("kind", ["cc", "gl"])
def test_t42_is_accepted_on_a_grid_that_represents_it(
    kind: Literal["cc", "gl"],
) -> None:
    grid = sg.clenshaw_curtis_grid(44, 85) if kind == "cc" else sg.gaussian_grid(43, 85)
    field = scalar_field(grid)

    result = sg.filter(field, "T42")

    np.testing.assert_allclose(result, field, rtol=2.0e-11, atol=2.0e-11)


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

    np.testing.assert_allclose(tapered, 0.1 * field, atol=2.0e-11)


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

    np.testing.assert_allclose(result, expected, rtol=2.0e-11, atol=2.0e-11)
    np.testing.assert_allclose(result.lat, target.latitude)
    np.testing.assert_allclose(result.lon, target.longitude)


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

    np.testing.assert_allclose(result.u, expected_u, rtol=2.0e-11, atol=2.0e-11)
    np.testing.assert_allclose(result.v, expected_v, rtol=2.0e-11, atol=2.0e-11)
    assert result.u.attrs == u.attrs
    assert result.v.attrs == v.attrs


def test_vector_regridding_selection_taper_and_accessor_paths() -> None:
    source = supported_grid("cc", latitude_order="descending", lon0=-180.0)
    target = supported_grid("gl", latitude_order="descending", lon0=-180.0)
    u, v = _axisymmetric_vector_field(source)
    latitude = np.deg2rad(source.latitude)[:, None]
    degree_two_u = 3.0 * np.sin(latitude) * np.cos(latitude)
    degree_two_v = 2.0 * np.sin(latitude) * np.cos(latitude)
    u = u.copy(data=degree_two_u * np.ones((1, source.nlon)))
    v = v.copy(data=degree_two_v * np.ones((1, source.nlon)))

    direct = sg.regrid_vector(u, v, target, spectral="T2", taper=0.1)
    dataarray_accessor = u.sg.regrid_vector(v, target, spectral="T2", taper=0.1)
    dataset_accessor = xr.Dataset({"u": u, "v": v}).sg.regrid_vector(
        target,
        spectral="T2",
        taper=0.1,
    )
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

    np.testing.assert_allclose(direct.u, expected_u, atol=2.0e-11)
    np.testing.assert_allclose(direct.v, expected_v, atol=2.0e-11)
    xr.testing.assert_identical(dataarray_accessor, direct)
    xr.testing.assert_identical(dataset_accessor, direct)
    np.testing.assert_allclose(direct.lat, target.latitude)
    np.testing.assert_allclose(direct.lon, target.longitude)


def test_vector_regridding_round_trip_is_band_limited() -> None:
    source = supported_grid("cc")
    intermediate = supported_grid("gl")
    u, v = _axisymmetric_vector_field(source)

    regridded = sg.regrid_vector(u, v, intermediate, spectral="T2")
    restored = sg.regrid_vector(
        regridded.u,
        regridded.v,
        source,
        spectral="T2",
    )

    np.testing.assert_allclose(restored.u, u, rtol=2.0e-11, atol=2.0e-11)
    np.testing.assert_allclose(restored.v, v, rtol=2.0e-11, atol=2.0e-11)


def test_combined_filter_and_regrid_matches_single_spectral_cycle_result() -> None:
    source = scalar_field(supported_grid("cc"))
    target = supported_grid("gl")

    result = sg.regrid(source, target, spectral="T2", taper=1.0)
    expected = sg.regrid(sg.filter(source, "T2"), target, spectral="T2")

    np.testing.assert_allclose(result, expected, rtol=2.0e-11, atol=2.0e-11)


def test_regridding_restores_descending_and_shifted_target_coordinates() -> None:
    source_grid = supported_grid("cc", latitude_order="descending", lon0=-180.0)
    target_grid = sg.gaussian_grid(
        16,
        36,
        latitude_order="descending",
        lon0=-180.0,
    )
    source = scalar_field(source_grid)

    result = sg.regrid(source, target_grid, spectral="T3")
    expected = scalar_field(target_grid)

    np.testing.assert_allclose(result, expected, rtol=2.0e-11, atol=2.0e-11)
    np.testing.assert_allclose(result.lat, expected.lat)
    np.testing.assert_allclose(result.lon, expected.lon)
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
    np.testing.assert_allclose(gradient.gradient_eastward, 0.0, atol=2.0e-13)
    np.testing.assert_allclose(
        gradient.gradient_northward, expected_north, atol=2.0e-13
    )
    np.testing.assert_allclose(laplacian, -2.0 * field / radius**2, atol=2.0e-24)
    # The inverse multiplier is O(R**2), so double-precision coefficient
    # roundoff appears as roughly 1e-3 in the exact nodal zero at the CC
    # equator.  The nonzero field scale is O(1e13).
    np.testing.assert_allclose(
        inverse,
        -(radius**2) * field / 2.0,
        rtol=2.0e-11,
        atol=2.0e-3,
    )


@pytest.mark.parametrize("kind", ["cc", "gl"])
def test_inverse_laplacian_uses_zero_mean_solution(kind: Literal["cc", "gl"]) -> None:
    grid = supported_grid(kind)
    field = degree_one_field(grid) + 7.0

    inverse = sg.inverse_laplacian(field)
    restored = sg.laplacian(inverse)

    np.testing.assert_allclose(restored, degree_one_field(grid), atol=2.0e-11)


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


def test_accessor_and_direct_scalar_paths_are_identical() -> None:
    field = scalar_field(supported_grid("cc"), leading=True)
    target = supported_grid("gl")

    direct = sg.regrid(field, target, spectral="T2", taper=0.4)
    accessor = field.sg.regrid(target, spectral="T2", taper=0.4)

    xr.testing.assert_identical(direct, accessor)


def test_scalar_operator_accessors_delegate_to_direct_functions() -> None:
    field = degree_one_field(supported_grid("cc"))

    xr.testing.assert_identical(field.sg.gradient(), sg.gradient(field))
    xr.testing.assert_identical(field.sg.laplacian(), sg.laplacian(field))
    xr.testing.assert_identical(
        field.sg.inverse_laplacian(), sg.inverse_laplacian(field)
    )


def test_spectral_error_cases_are_explicit() -> None:
    field = scalar_field(supported_grid("cc"))

    with pytest.raises(ValueError, match="either spectral"):
        sg.filter(field, "T2", lmin=0, lmax=2)
    with pytest.raises(ValueError, match="both lmin"):
        sg.filter(field, lmax=2)
    with pytest.raises(ValueError, match="taper"):
        sg.filter(field, "T2", taper=0.0)
    with pytest.raises(ValueError, match="exceeds"):
        sg.filter(field, "T99")
    with pytest.raises(ValueError, match="exceeds"):
        sg.regrid(field, sg.clenshaw_curtis_grid(9, 18), spectral="T8")
    with pytest.raises(ValueError, match="must be distinct"):
        sg.gradient(field, eastward="gradient", northward="gradient")
