"""DataArray xarray accessor delegation contracts."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
import xarray as xr

import spharmgrid as sg
from tests.conftest import (
    degree_one_field,
    scalar_field,
    solid_body_wind,
    supported_grid,
)


def _assert_dataset_identical(actual: xr.Dataset, expected: xr.Dataset) -> None:
    """Check Dataset component order, metadata, and values."""
    assert tuple(actual.data_vars) == tuple(expected.data_vars)
    xr.testing.assert_identical(actual, expected)


@pytest.mark.parametrize("kind", ["cc", "gl"])
def test_dataarray_grid_accessor_properties(kind: Literal["cc", "gl"]) -> None:
    grid = supported_grid(kind)
    field = scalar_field(grid)
    expected = sg.detect_grid(field)

    assert field.sg.grid.kind == expected.kind
    np.testing.assert_array_equal(field.sg.grid.latitude, expected.latitude)
    np.testing.assert_array_equal(field.sg.grid.longitude, expected.longitude)
    assert field.sg.grid_type == expected.kind


def test_dataarray_scalar_accessors() -> None:
    field = scalar_field(supported_grid("cc"), leading=True)
    target = supported_grid("gl")
    radius = 1.1 * sg.EARTH_RADIUS_M

    xr.testing.assert_identical(
        field.sg.filter("T2", taper=0.4),
        sg.filter(field, "T2", taper=0.4),
    )
    xr.testing.assert_identical(
        field.sg.regrid(target, truncation="T2", taper=0.4),
        sg.regrid(field, target, truncation="T2", taper=0.4),
    )
    xr.testing.assert_identical(
        field.sg.laplacian(radius=radius),
        sg.laplacian(field, radius=radius),
    )
    xr.testing.assert_identical(
        field.sg.inverse_laplacian(radius=radius),
        sg.inverse_laplacian(field, radius=radius),
    )


def test_dataarray_scalar_to_vector_accessors() -> None:
    grid = supported_grid("cc")
    field = degree_one_field(grid)
    radius = 1.1 * sg.EARTH_RADIUS_M
    u, v = solid_body_wind(grid)

    expected_gradient = sg.gradient(
        field,
        eastward="east_gradient",
        northward="north_gradient",
        radius=radius,
    )
    actual_gradient = field.sg.gradient(
        eastward="east_gradient",
        northward="north_gradient",
        radius=radius,
    )
    _assert_dataset_identical(actual_gradient, expected_gradient)

    vorticity = sg.vorticity(u, v)
    expected_rotational = sg.rotational_wind(
        vorticity,
        eastward="rot_u",
        northward="rot_v",
        radius=radius,
    )
    actual_rotational = vorticity.sg.rotational_wind(
        eastward="rot_u",
        northward="rot_v",
        radius=radius,
    )
    _assert_dataset_identical(actual_rotational, expected_rotational)

    divergence = sg.divergence(u, v)
    expected_divergent = sg.divergent_wind(
        divergence,
        eastward="div_u",
        northward="div_v",
        radius=radius,
    )
    actual_divergent = divergence.sg.divergent_wind(
        eastward="div_u",
        northward="div_v",
        radius=radius,
    )
    _assert_dataset_identical(actual_divergent, expected_divergent)

    diagnostics = sg.kinematics(u, v)
    expected_wind = sg.wind(
        diagnostics.vo,
        diagnostics.d,
        eastward="wind_u",
        northward="wind_v",
        radius=radius,
    )
    actual_wind = diagnostics.vo.sg.wind(
        diagnostics.d,
        eastward="wind_u",
        northward="wind_v",
        radius=radius,
    )
    _assert_dataset_identical(actual_wind, expected_wind)

    potentials = sg.potentials(u, v)
    expected_wind = sg.wind(
        potentials.strf,
        potentials.vp,
        eastward="potential_u",
        northward="potential_v",
        radius=radius,
    )
    actual_wind = potentials.strf.sg.wind(
        potentials.vp,
        eastward="potential_u",
        northward="potential_v",
        radius=radius,
    )
    _assert_dataset_identical(actual_wind, expected_wind)


def test_dataarray_regrid_vector_accessor() -> None:
    source = supported_grid("cc", latitude_order="descending", lon0=-180.0)
    target = supported_grid("gl", latitude_order="descending", lon0=-180.0)
    u, v = solid_body_wind(source)

    expected = sg.regrid_vector(
        u,
        v,
        target,
        truncation="T2",
        taper=0.1,
        eastward="eastward",
        northward="northward",
    )
    actual = u.sg.regrid_vector(
        v,
        target,
        truncation="T2",
        taper=0.1,
        eastward="eastward",
        northward="northward",
    )

    _assert_dataset_identical(actual, expected)


def test_dataarray_vector_accessors() -> None:
    grid = supported_grid("cc")
    u, v = solid_body_wind(grid)
    radius = 1.1 * sg.EARTH_RADIUS_M

    expected = sg.vorticity(u, v, output="relative_vorticity", radius=radius)
    actual = u.sg.vorticity(v, output="relative_vorticity", radius=radius)
    xr.testing.assert_identical(actual, expected)

    expected = sg.divergence(u, v, output="wind_divergence", radius=radius)
    actual = u.sg.divergence(v, output="wind_divergence", radius=radius)
    xr.testing.assert_identical(actual, expected)

    expected = sg.kinematics(
        u,
        v,
        vorticity="relative_vorticity",
        divergence="wind_divergence",
        radius=radius,
    )
    actual = u.sg.kinematics(
        v,
        vorticity="relative_vorticity",
        divergence="wind_divergence",
        radius=radius,
    )
    _assert_dataset_identical(actual, expected)

    expected = sg.streamfunction(u, v, output="streamfunction", radius=radius)
    actual = u.sg.streamfunction(v, output="streamfunction", radius=radius)
    xr.testing.assert_identical(actual, expected)

    expected = sg.velocity_potential(u, v, output="velocity_potential", radius=radius)
    actual = u.sg.velocity_potential(v, output="velocity_potential", radius=radius)
    xr.testing.assert_identical(actual, expected)

    expected = sg.potentials(
        u,
        v,
        streamfunction="streamfunction",
        velocity_potential="velocity_potential",
        radius=radius,
    )
    actual = u.sg.potentials(
        v,
        streamfunction="streamfunction",
        velocity_potential="velocity_potential",
        radius=radius,
    )
    _assert_dataset_identical(actual, expected)

    expected = sg.helmholtz(
        u,
        v,
        divergent_eastward="divergent_u",
        divergent_northward="divergent_v",
        rotational_eastward="rotational_u",
        rotational_northward="rotational_v",
        radius=radius,
    )
    actual = u.sg.helmholtz(
        v,
        divergent_eastward="divergent_u",
        divergent_northward="divergent_v",
        rotational_eastward="rotational_u",
        rotational_northward="rotational_v",
        radius=radius,
    )
    _assert_dataset_identical(actual, expected)

    expected = sg.vector_laplacian(
        u,
        v,
        eastward="laplacian_u",
        northward="laplacian_v",
        radius=radius,
    )
    actual = u.sg.vector_laplacian(
        v,
        eastward="laplacian_u",
        northward="laplacian_v",
        radius=radius,
    )
    _assert_dataset_identical(actual, expected)

    laplacian = sg.vector_laplacian(u, v)
    expected = sg.inverse_vector_laplacian(
        laplacian.u,
        laplacian.v,
        eastward="restored_u",
        northward="restored_v",
        radius=radius,
    )
    actual = laplacian.u.sg.inverse_vector_laplacian(
        laplacian.v,
        eastward="restored_u",
        northward="restored_v",
        radius=radius,
    )
    _assert_dataset_identical(actual, expected)

    scalar = degree_one_field(grid)
    gradient = sg.gradient(scalar, radius=radius)
    expected = sg.inverse_gradient(
        gradient.gradient_eastward,
        gradient.gradient_northward,
        output="potential",
        radius=radius,
    )
    actual = gradient.gradient_eastward.sg.inverse_gradient(
        gradient.gradient_northward,
        output="potential",
        radius=radius,
    )
    xr.testing.assert_identical(actual, expected)


def test_dataarray_accessor_preserves_leading_dimensions() -> None:
    grid = supported_grid("cc")
    u, v = solid_body_wind(grid)
    member = xr.DataArray(["first", "second"], dims="member", name="member")
    u = xr.concat([u, 2.0 * u], dim=member)
    v = xr.concat([v, 2.0 * v], dim=member)

    expected = sg.kinematics(u, v)
    actual = u.sg.kinematics(v)

    _assert_dataset_identical(actual, expected)
    assert actual.vo.dims == ("member", "lat", "lon")
    np.testing.assert_array_equal(actual.member.values, member.values)
