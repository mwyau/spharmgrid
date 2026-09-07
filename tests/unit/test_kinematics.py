"""Analytic wind diagnostics and inverse vector-transform identities."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
import xarray as xr

import spharmgrid as sg
from tests.conftest import (
    degree_one_field,
    solid_body_wind,
    supported_grid,
)


@pytest.mark.parametrize("kind", ["cc", "gl"])
def test_solid_body_rotation_has_known_vorticity_and_zero_divergence(
    kind: Literal["cc", "gl"],
) -> None:
    grid = supported_grid(kind)
    u, v = solid_body_wind(grid)
    amplitude = 10.0

    diagnostics = sg.kinematics(u, v)
    expected_vorticity = (
        2.0
        * amplitude
        * np.sin(np.deg2rad(grid.latitude))[:, None]
        / sg.EARTH_RADIUS_M
        * np.ones((1, grid.nlon))
    )

    np.testing.assert_allclose(
        diagnostics.vo, expected_vorticity, rtol=0.0, atol=1.0e-19
    )
    np.testing.assert_allclose(diagnostics.d, 0.0, rtol=0.0, atol=0.0)
    assert diagnostics.vo.attrs["standard_name"] == "atmosphere_relative_vorticity"
    assert diagnostics.d.attrs["standard_name"] == "divergence_of_wind"


@pytest.mark.parametrize("kind", ["cc", "gl"])
def test_meridional_degree_one_flow_has_known_divergence_and_zero_vorticity(
    kind: Literal["cc", "gl"],
) -> None:
    grid = supported_grid(kind)
    shape = (grid.nlat, grid.nlon)
    u = xr.DataArray(
        np.zeros(shape),
        dims=("lat", "lon"),
        coords={"lat": grid.latitude, "lon": grid.longitude},
        name="u",
    )
    amplitude = 10.0
    v = xr.DataArray(
        amplitude
        * np.cos(np.deg2rad(grid.latitude))[:, None]
        * np.ones((1, grid.nlon)),
        dims=("lat", "lon"),
        coords={"lat": grid.latitude, "lon": grid.longitude},
        name="v",
    )

    diagnostics = sg.kinematics(u, v)
    expected_divergence = (
        -2.0
        * amplitude
        * np.sin(np.deg2rad(grid.latitude))[:, None]
        / sg.EARTH_RADIUS_M
        * np.ones((1, grid.nlon))
    )

    np.testing.assert_allclose(diagnostics.vo, 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        diagnostics.d, expected_divergence, rtol=0.0, atol=1.0e-19
    )


@pytest.mark.parametrize("kind", ["cc", "gl"])
def test_potentials_and_inverse_wind_round_trip(kind: Literal["cc", "gl"]) -> None:
    grid = supported_grid(kind)
    scalar = degree_one_field(grid)
    streamfunction = scalar * (-3.0 * sg.EARTH_RADIUS_M)
    velocity_potential = scalar * (2.0 * sg.EARTH_RADIUS_M)
    streamfunction.name = "strf"
    velocity_potential.name = "vp"

    rotational = sg.rotational_wind(streamfunction)
    divergent = sg.divergent_wind(velocity_potential)
    u = rotational.u_rotational + divergent.u_divergent
    v = rotational.v_rotational + divergent.v_divergent
    u.name = "u"
    v.name = "v"

    diagnostics = sg.kinematics(u, v)
    recovered_potentials = sg.potentials(u, v)
    reconstructed = sg.wind(diagnostics.vo, diagnostics.d)

    np.testing.assert_allclose(
        recovered_potentials.strf,
        streamfunction,
        rtol=0.0,
        atol=3.0e-7,
    )
    np.testing.assert_allclose(
        recovered_potentials.vp,
        velocity_potential,
        rtol=0.0,
        atol=1.0e-7,
    )
    np.testing.assert_allclose(reconstructed.u, u, rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(reconstructed.v, v, rtol=0.0, atol=1.0e-14)
    assert (
        recovered_potentials.strf.attrs["standard_name"]
        == "atmosphere_horizontal_streamfunction"
    )
    assert (
        recovered_potentials.vp.attrs["standard_name"]
        == "atmosphere_horizontal_velocity_potential"
    )
    assert reconstructed.u.attrs["standard_name"] == "eastward_wind"
    assert reconstructed.v.attrs["standard_name"] == "northward_wind"


@pytest.mark.parametrize("kind", ["cc", "gl"])
def test_non_zonal_potential_round_trip_restores_cyclic_coordinates(
    kind: Literal["cc", "gl"],
) -> None:
    grid = supported_grid(kind, latitude_order="descending", lon0=-180.0)
    latitude = np.deg2rad(grid.latitude)[:, None]
    longitude = np.deg2rad(grid.longitude)[None, :]
    coordinates = {"lat": grid.latitude, "lon": grid.longitude}
    streamfunction = xr.DataArray(
        3.0 * sg.EARTH_RADIUS_M * np.cos(latitude) * np.cos(longitude),
        dims=("lat", "lon"),
        coords=coordinates,
        name="strf",
    )
    velocity_potential = xr.DataArray(
        2.0 * sg.EARTH_RADIUS_M * np.cos(latitude) * np.sin(longitude),
        dims=("lat", "lon"),
        coords=coordinates,
        name="vp",
    )

    reconstructed = sg.wind(streamfunction, velocity_potential)
    recovered = sg.potentials(reconstructed.u, reconstructed.v)

    np.testing.assert_allclose(recovered.strf, streamfunction, rtol=0.0, atol=1.0e-7)
    np.testing.assert_allclose(recovered.vp, velocity_potential, rtol=0.0, atol=1.0e-7)
    xr.testing.assert_identical(reconstructed.lat, streamfunction.lat)
    xr.testing.assert_identical(reconstructed.lon, streamfunction.lon)


@pytest.mark.parametrize("kind", ["cc", "gl"])
def test_rotational_and_divergent_parts_satisfy_cross_diagnostic_identities(
    kind: Literal["cc", "gl"],
) -> None:
    grid = supported_grid(kind)
    scalar = degree_one_field(grid)
    vo = sg.laplacian(scalar * sg.EARTH_RADIUS_M**2)
    vo.name = "vo"
    divergence = sg.laplacian(scalar * sg.EARTH_RADIUS_M**2)
    divergence.name = "d"

    rotational = sg.rotational_wind(vo)
    divergent = sg.divergent_wind(divergence)

    rotational_diagnostics = sg.kinematics(
        rotational.u_rotational, rotational.v_rotational
    )
    divergent_diagnostics = sg.kinematics(divergent.u_divergent, divergent.v_divergent)

    np.testing.assert_allclose(rotational_diagnostics.d, 0.0, rtol=0.0, atol=1.0e-30)
    np.testing.assert_allclose(divergent_diagnostics.vo, 0.0, rtol=0.0, atol=1.0e-30)
    np.testing.assert_allclose(rotational_diagnostics.vo, vo, rtol=0.0, atol=5.0e-14)
    np.testing.assert_allclose(
        divergent_diagnostics.d, divergence, rtol=0.0, atol=5.0e-14
    )
    assert "standard_name" not in rotational.u_rotational.attrs
    assert "standard_name" not in divergent.v_divergent.attrs


@pytest.mark.parametrize("kind", ["cc", "gl"])
def test_helmholtz_separates_known_rotational_and_divergent_degree_one_flows(
    kind: Literal["cc", "gl"],
) -> None:
    grid = supported_grid(kind)
    latitude = np.deg2rad(grid.latitude)[:, None]
    shape = (grid.nlat, grid.nlon)
    coordinates = {"lat": grid.latitude, "lon": grid.longitude}
    rotational_u = xr.DataArray(
        10.0 * np.cos(latitude) * np.ones((1, grid.nlon)),
        dims=("lat", "lon"),
        coords=coordinates,
    )
    rotational_v = xr.DataArray(
        np.zeros(shape),
        dims=("lat", "lon"),
        coords=coordinates,
    )
    divergent_u = xr.DataArray(
        np.zeros(shape),
        dims=("lat", "lon"),
        coords=coordinates,
    )
    divergent_v = xr.DataArray(
        7.0 * np.cos(latitude) * np.ones((1, grid.nlon)),
        dims=("lat", "lon"),
        coords=coordinates,
    )
    u = (rotational_u + divergent_u).rename("u")
    v = (rotational_v + divergent_v).rename("v")

    result = sg.helmholtz(u, v)

    np.testing.assert_allclose(
        result.u_rotational, rotational_u, rtol=0.0, atol=3.0e-14
    )
    np.testing.assert_allclose(
        result.v_rotational, rotational_v, rtol=0.0, atol=1.0e-30
    )
    np.testing.assert_allclose(result.u_divergent, divergent_u, rtol=0.0, atol=1.0e-30)
    np.testing.assert_allclose(result.v_divergent, divergent_v, rtol=0.0, atol=3.0e-14)
    np.testing.assert_allclose(
        result.u_divergent + result.u_rotational,
        u,
        rtol=0.0,
        atol=3.0e-14,
    )
    np.testing.assert_allclose(
        result.v_divergent + result.v_rotational,
        v,
        rtol=0.0,
        atol=3.0e-14,
    )
    assert "standard_name" not in result.u_divergent.attrs
    assert result.u_rotational.attrs["long_name"] == "Eastward rotational wind"


@pytest.mark.parametrize("kind", ["cc", "gl"])
def test_inverse_gradient_projects_pure_rotation_to_zero(
    kind: Literal["cc", "gl"],
) -> None:
    grid = supported_grid(kind)
    latitude = np.deg2rad(grid.latitude)[:, None]
    coordinates = {"lat": grid.latitude, "lon": grid.longitude}
    eastward = xr.DataArray(
        10.0 * np.cos(latitude) * np.ones((1, grid.nlon)) / sg.EARTH_RADIUS_M,
        dims=("lat", "lon"),
        coords=coordinates,
        attrs={"units": "K m-1"},
    )
    northward = xr.DataArray(
        np.zeros((grid.nlat, grid.nlon)),
        dims=("lat", "lon"),
        coords=coordinates,
        attrs={"units": "K m-1"},
    )

    assert np.max(np.abs(eastward.values)) > 1.0e-8

    recovered = sg.inverse_gradient(eastward, northward)

    np.testing.assert_allclose(recovered, 0.0, rtol=0.0, atol=5.0e-14)


@pytest.mark.parametrize("kind", ["cc", "gl"])
def test_inverse_gradient_recovers_known_potential_from_mixed_vector_field(
    kind: Literal["cc", "gl"],
) -> None:
    grid = supported_grid(kind)
    latitude = np.deg2rad(grid.latitude)[:, None]
    coordinates = {"lat": grid.latitude, "lon": grid.longitude}
    eastward = xr.DataArray(
        10.0 * np.cos(latitude) * np.ones((1, grid.nlon)) / sg.EARTH_RADIUS_M,
        dims=("lat", "lon"),
        coords=coordinates,
        attrs={"units": "K m-1"},
    )
    northward = xr.DataArray(
        np.cos(latitude) * np.ones((1, grid.nlon)) / sg.EARTH_RADIUS_M,
        dims=("lat", "lon"),
        coords=coordinates,
        attrs={"units": "K m-1"},
    )

    projected = sg.inverse_gradient(
        eastward,
        northward,
        output="potential",
    )

    expected = np.sin(latitude) * np.ones((1, grid.nlon))
    np.testing.assert_allclose(projected, expected, rtol=0.0, atol=5.0e-14)
    assert projected.name == "potential"
    assert projected.attrs["units"] == "K"
    assert "standard_name" not in projected.attrs


@pytest.mark.parametrize("kind", ["cc", "gl"])
def test_vector_laplacian_has_spherepack_degree_one_eigenvalue(
    kind: Literal["cc", "gl"],
) -> None:
    grid = supported_grid(kind)
    u, _ = solid_body_wind(grid)
    latitude = np.deg2rad(grid.latitude)[:, None]
    v = xr.DataArray(
        7.0 * np.cos(latitude) * np.ones((1, grid.nlon)),
        dims=("lat", "lon"),
        coords={"lat": grid.latitude, "lon": grid.longitude},
        name="v",
        attrs={"standard_name": "northward_wind", "units": "m s-1"},
    )
    u.attrs = {"standard_name": "eastward_wind", "units": "m s-1"}

    laplacian = sg.vector_laplacian(u, v)
    restored = sg.inverse_vector_laplacian(laplacian.u, laplacian.v)
    inverse = sg.inverse_vector_laplacian(u, v)
    inverse_restored = sg.vector_laplacian(inverse.u, inverse.v)
    eigenvalue = -2.0 / sg.EARTH_RADIUS_M**2

    np.testing.assert_allclose(laplacian.u, eigenvalue * u, rtol=0.0, atol=2.0e-25)
    np.testing.assert_allclose(laplacian.v, eigenvalue * v, rtol=0.0, atol=1.0e-25)
    np.testing.assert_allclose(restored.u, u, rtol=0.0, atol=5.0e-14)
    np.testing.assert_allclose(restored.v, v, rtol=0.0, atol=3.0e-14)
    assert "standard_name" not in laplacian.u.attrs
    assert laplacian.u.attrs["units"] == "m s-1 m-2"
    assert restored.u.attrs["units"] == "m s-1"
    assert restored.v.attrs["units"] == "m s-1"
    assert inverse.u.attrs["units"] == "m s-1 m2"
    assert inverse.v.attrs["units"] == "m s-1 m2"
    assert inverse_restored.u.attrs["units"] == "m s-1"
    assert inverse_restored.v.attrs["units"] == "m s-1"


@pytest.mark.parametrize("kind", ["cc", "gl"])
def test_new_vector_operations_restore_descending_shifted_coordinates(
    kind: Literal["cc", "gl"],
) -> None:
    grid = supported_grid(kind, latitude_order="descending", lon0=-180.0)
    latitude = np.deg2rad(grid.latitude)[:, None]
    longitude = np.deg2rad(grid.longitude)[None, :]
    coordinates = {"lat": grid.latitude, "lon": grid.longitude}
    streamfunction = xr.DataArray(
        3.0 * sg.EARTH_RADIUS_M * np.cos(latitude) * np.cos(longitude),
        dims=("lat", "lon"),
        coords=coordinates,
        name="strf",
    )
    velocity_potential = xr.DataArray(
        2.0 * sg.EARTH_RADIUS_M * np.cos(latitude) * np.sin(longitude),
        dims=("lat", "lon"),
        coords=coordinates,
        name="vp",
    )
    wind = sg.wind(streamfunction, velocity_potential)
    components = sg.helmholtz(wind.u, wind.v)
    laplacian = sg.vector_laplacian(wind.u, wind.v)
    restored = sg.inverse_vector_laplacian(laplacian.u, laplacian.v)
    scalar = xr.DataArray(
        4.0 + np.cos(latitude) * np.cos(longitude),
        dims=("lat", "lon"),
        coords=coordinates,
    )
    gradient = sg.gradient(scalar)
    recovered = sg.inverse_gradient(
        gradient.gradient_eastward,
        gradient.gradient_northward,
    )

    np.testing.assert_allclose(
        components.u_divergent + components.u_rotational,
        wind.u,
        rtol=0.0,
        atol=5.0e-14,
    )
    np.testing.assert_allclose(
        components.v_divergent + components.v_rotational,
        wind.v,
        rtol=0.0,
        atol=5.0e-14,
    )
    np.testing.assert_allclose(restored.u, wind.u, rtol=0.0, atol=5.0e-14)
    np.testing.assert_allclose(restored.v, wind.v, rtol=0.0, atol=5.0e-14)
    np.testing.assert_allclose(
        recovered,
        scalar - 4.0,
        rtol=0.0,
        atol=5.0e-14,
    )
    for result in (
        components.u_divergent,
        components.v_rotational,
        laplacian.u,
        restored.v,
        recovered,
    ):
        xr.testing.assert_identical(result.lat, wind.lat)
        xr.testing.assert_identical(result.lon, wind.lon)
