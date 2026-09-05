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

    np.testing.assert_allclose(diagnostics.vo, expected_vorticity, atol=2.0e-15)
    np.testing.assert_allclose(diagnostics.d, 0.0, atol=2.0e-15)
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

    np.testing.assert_allclose(diagnostics.vo, 0.0, atol=2.0e-15)
    np.testing.assert_allclose(diagnostics.d, expected_divergence, atol=2.0e-15)


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
        rtol=2.0e-11,
        atol=2.0e-5,
    )
    np.testing.assert_allclose(
        recovered_potentials.vp,
        velocity_potential,
        rtol=2.0e-11,
        atol=2.0e-5,
    )
    np.testing.assert_allclose(reconstructed.u, u, atol=2.0e-11)
    np.testing.assert_allclose(reconstructed.v, v, atol=2.0e-11)
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

    np.testing.assert_allclose(
        recovered.strf, streamfunction, rtol=2.0e-11, atol=2.0e-5
    )
    np.testing.assert_allclose(
        recovered.vp, velocity_potential, rtol=2.0e-11, atol=2.0e-5
    )
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

    np.testing.assert_allclose(rotational_diagnostics.d, 0.0, atol=2.0e-15)
    np.testing.assert_allclose(divergent_diagnostics.vo, 0.0, atol=2.0e-15)
    # The CC equator is an exact analytic zero; two spin/scalar transforms
    # leave roundoff at several e-15 there.
    np.testing.assert_allclose(rotational_diagnostics.vo, vo, atol=1.0e-14)
    np.testing.assert_allclose(divergent_diagnostics.d, divergence, atol=1.0e-14)
    assert "standard_name" not in rotational.u_rotational.attrs
    assert "standard_name" not in divergent.v_divergent.attrs


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
    np.testing.assert_allclose(result.u, u, atol=2.0e-11)

    ambiguous_vorticity = both.assign(other_vo=kin.vo.rename("other_vo"))
    result = ambiguous_vorticity.sg.wind(source="potentials")
    np.testing.assert_allclose(result.u, u, atol=2.0e-11)
