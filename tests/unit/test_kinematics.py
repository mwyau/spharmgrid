"""Analytic wind diagnostics and inverse vector-transform identities."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
import xarray as xr

import spharmgrid as sg
from tests.conftest import degree_one_field, solid_body_wind, supported_grid


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
        atol=1.0e-7,
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
def test_helmholtz_recovers_analytic_rotational_and_divergent_parts(
    kind: Literal["cc", "gl"],
) -> None:
    grid = supported_grid(kind)
    scalar = degree_one_field(grid)
    streamfunction = (3.0 * sg.EARTH_RADIUS_M * scalar).rename("strf")
    velocity_potential = (-2.0 * sg.EARTH_RADIUS_M * scalar).rename("vp")
    rotational = sg.rotational_wind(streamfunction)
    divergent = sg.divergent_wind(velocity_potential)
    u = (rotational.u_rotational + divergent.u_divergent).rename("u")
    v = (rotational.v_rotational + divergent.v_divergent).rename("v")

    result = sg.helmholtz(u, v)

    np.testing.assert_allclose(
        result.u_divergent, divergent.u_divergent, rtol=0.0, atol=1.0e-30
    )
    np.testing.assert_allclose(
        result.v_divergent, divergent.v_divergent, rtol=0.0, atol=1.0e-14
    )
    np.testing.assert_allclose(
        result.u_rotational, rotational.u_rotational, rtol=0.0, atol=1.0e-14
    )
    np.testing.assert_allclose(
        result.v_rotational, rotational.v_rotational, rtol=0.0, atol=1.0e-30
    )
    np.testing.assert_allclose(
        result.u_divergent + result.u_rotational,
        u,
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        result.v_divergent + result.v_rotational,
        v,
        rtol=0.0,
        atol=1.0e-14,
    )
    assert "standard_name" not in result.u_divergent.attrs
    assert result.u_rotational.attrs["long_name"] == "Eastward rotational wind"


@pytest.mark.parametrize("kind", ["cc", "gl"])
def test_inverse_gradient_recovers_irrotational_potential_and_zero_mode(
    kind: Literal["cc", "gl"],
) -> None:
    grid = supported_grid(kind)
    field = degree_one_field(grid) + 4.0
    field.attrs["units"] = "K"
    gradient = sg.gradient(field)
    rotational_u, rotational_v = solid_body_wind(grid)
    projected = sg.inverse_gradient(
        gradient.gradient_eastward + rotational_u / sg.EARTH_RADIUS_M,
        gradient.gradient_northward + rotational_v / sg.EARTH_RADIUS_M,
        output="potential",
    )

    expected = degree_one_field(grid)
    np.testing.assert_allclose(projected, expected, rtol=0.0, atol=3.0e-14)
    recovered_gradient = sg.gradient(projected)
    np.testing.assert_allclose(
        recovered_gradient.gradient_eastward,
        gradient.gradient_eastward,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        recovered_gradient.gradient_northward,
        gradient.gradient_northward,
        rtol=0.0,
        atol=1.0e-20,
    )
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
        atol=3.0e-14,
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


def test_dataset_discovery_output_overrides_and_direct_accessor_equivalence() -> None:
    grid = supported_grid("cc")
    u, v = solid_body_wind(grid)
    dataset = xr.Dataset({"ua": u, "va": v})
    dataset.ua.attrs["standard_name"] = "eastward_wind"
    dataset.va.attrs["standard_name"] = "northward_wind"

    accessor = dataset.sg.kinematics(vorticity="vort", divergence="div")
    direct = sg.kinematics(dataset.ua, dataset.va, vorticity="vort", divergence="div")

    xr.testing.assert_identical(accessor, direct)
    assert accessor.vort.attrs["standard_name"] == "atmosphere_relative_vorticity"
    assert accessor.div.attrs["standard_name"] == "divergence_of_wind"


def test_dataarray_accessor_matches_direct_path_with_leading_dimensions() -> None:
    grid = supported_grid("cc")
    u, v = solid_body_wind(grid)
    member = xr.DataArray(["first", "second"], dims="member", name="member")
    u = xr.concat([u, 2.0 * u], dim=member)
    v = xr.concat([v, 2.0 * v], dim=member)

    direct = sg.kinematics(u, v)
    accessor = u.sg.kinematics(v)

    xr.testing.assert_identical(accessor, direct)
    assert direct.vo.dims == ("member", "lat", "lon")
    np.testing.assert_array_equal(direct.member.values, member.values)


def test_dataarray_inverse_wind_accessors_delegate_to_direct_functions() -> None:
    grid = supported_grid("cc")
    u, v = solid_body_wind(grid)
    diagnostics = sg.kinematics(u, v)
    potential = sg.potentials(u, v)

    xr.testing.assert_identical(
        diagnostics.vo.sg.rotational_wind(), sg.rotational_wind(diagnostics.vo)
    )
    xr.testing.assert_identical(
        diagnostics.d.sg.divergent_wind(), sg.divergent_wind(diagnostics.d)
    )
    xr.testing.assert_identical(
        diagnostics.vo.sg.wind(diagnostics.d), sg.wind(diagnostics.vo, diagnostics.d)
    )
    xr.testing.assert_identical(
        potential.strf.sg.wind(potential.vp), sg.wind(potential.strf, potential.vp)
    )


def test_new_vector_accessors_delegate_to_direct_functions() -> None:
    grid = supported_grid("cc")
    u, v = solid_body_wind(grid)
    gradient = sg.gradient(degree_one_field(grid))
    dataset = xr.Dataset({"u": u, "v": v, **gradient.data_vars})

    xr.testing.assert_identical(u.sg.helmholtz(v), sg.helmholtz(u, v))
    xr.testing.assert_identical(dataset.sg.helmholtz(), sg.helmholtz(u, v))
    xr.testing.assert_identical(
        gradient.gradient_eastward.sg.inverse_gradient(gradient.gradient_northward),
        sg.inverse_gradient(gradient.gradient_eastward, gradient.gradient_northward),
    )
    xr.testing.assert_identical(
        dataset.sg.inverse_gradient(),
        sg.inverse_gradient(gradient.gradient_eastward, gradient.gradient_northward),
    )
    xr.testing.assert_identical(u.sg.vector_laplacian(v), sg.vector_laplacian(u, v))
    xr.testing.assert_identical(
        dataset.sg.vector_laplacian(), sg.vector_laplacian(u, v)
    )
    laplacian = sg.vector_laplacian(u, v)
    xr.testing.assert_identical(
        laplacian.u.sg.inverse_vector_laplacian(laplacian.v),
        sg.inverse_vector_laplacian(laplacian.u, laplacian.v),
    )


def test_dataset_discovery_reports_ambiguity_and_wind_source_ambiguity() -> None:
    grid = supported_grid("cc")
    u, v = solid_body_wind(grid)
    duplicate = u.rename("other_u")
    duplicate.attrs["standard_name"] = "eastward_wind"
    u.attrs["standard_name"] = "eastward_wind"
    dataset = xr.Dataset({"u": u, "other_u": duplicate, "v": v})

    with pytest.raises(ValueError, match="ambiguous 'u'"):
        dataset.sg.vorticity()

    kin = sg.kinematics(u, v)
    potential = sg.potentials(u, v)
    both = xr.Dataset(
        {"vo": kin.vo, "d": kin.d, "strf": potential.strf, "vp": potential.vp}
    )
    with pytest.raises(ValueError, match="both vorticity/divergence"):
        both.sg.wind()
    result = both.sg.wind(source="potentials")
    np.testing.assert_allclose(result.u, u, rtol=0.0, atol=5.0e-13)

    ambiguous_vorticity = both.assign(other_vo=kin.vo.rename("other_vo"))
    result = ambiguous_vorticity.sg.wind(source="potentials")
    np.testing.assert_allclose(result.u, u, rtol=0.0, atol=5.0e-13)
