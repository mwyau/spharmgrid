# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Analytic scalar, vector, and atmospheric-operator tests for JAX."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array, config

import spharmgrid.jax as sgj
from tests.optional.jax._fields import (
    pure_gradient_values,
    pure_rotational_values,
    scalar_potentials,
    vector_values,
)


def _as_dtype(values: np.ndarray, dtype: object) -> Array:
    if dtype == jnp.float64 and not config.read("jax_enable_x64"):
        pytest.skip("float64 validation requires JAX x64 enabled")
    return jnp.asarray(values, dtype=dtype)


def _assert_close(actual: object, expected: object) -> None:
    actual_values = np.asarray(actual)
    expected_values = np.broadcast_to(np.asarray(expected), actual_values.shape)
    np.testing.assert_allclose(
        actual_values,
        expected_values,
        rtol=2.0e-10,
        atol=2.0e-11,
    )


@pytest.mark.parametrize("grid_name", ["gl_grid", "cc_grid"])
@pytest.mark.parametrize("dtype", [jnp.float64])
def test_scalar_harmonic_identities(
    request: pytest.FixtureRequest,
    grid_name: str,
    dtype: object,
) -> None:
    grid = request.getfixturevalue(grid_name)
    latitude = np.deg2rad(grid.latitude)[:, None]
    longitude = np.deg2rad(grid.longitude)[None, :]
    sine = np.sin(latitude)
    cosine = np.cos(latitude)
    degree_one = sine
    degree_two_zonal = (3.0 * sine**2 - 1.0) / 2.0
    degree_two_sectoral = cosine**2 * np.cos(2.0 * longitude)
    field = np.broadcast_to(
        1.25 + 0.5 * degree_one + 0.75 * degree_two_zonal + 0.2 * degree_two_sectoral,
        (grid.nlat, grid.nlon),
    ).copy()
    radius = 2.5
    array = _as_dtype(field, dtype)

    low = sgj.filter(array, "T1", grid=grid)
    band = sgj.filter(array, "T2-2", grid=grid)
    explicit_band = sgj.filter(array, grid=grid, lmin=2, lmax=2)
    tapered = sgj.filter(
        _as_dtype(degree_two_sectoral, dtype),
        "T2",
        grid=grid,
        taper=0.2,
    )
    _assert_close(low, 1.25 + 0.5 * degree_one)
    _assert_close(
        band,
        0.75 * degree_two_zonal + 0.2 * degree_two_sectoral,
    )
    _assert_close(explicit_band, band)
    _assert_close(tapered, 0.2 * degree_two_sectoral)

    expected_laplacian = (
        -1.0 * degree_one - 4.5 * degree_two_zonal - 1.2 * degree_two_sectoral
    ) / radius**2
    expected_inverse = radius**2 * (
        -0.25 * degree_one
        - 0.125 * degree_two_zonal
        - (0.2 / 6.0) * degree_two_sectoral
    )
    _assert_close(sgj.laplacian(array, grid=grid, radius=radius), expected_laplacian)
    _assert_close(
        sgj.inverse_laplacian(array, grid=grid, radius=radius),
        expected_inverse,
    )


@pytest.mark.parametrize("grid_name", ["gl_grid", "cc_grid"])
@pytest.mark.parametrize("dtype", [jnp.float64])
def test_gradient_and_inverse_gradient_use_geographic_components(
    request: pytest.FixtureRequest,
    grid_name: str,
    dtype: object,
) -> None:
    grid = request.getfixturevalue(grid_name)
    latitude = np.deg2rad(grid.latitude)[:, None]
    longitude = np.deg2rad(grid.longitude)[None, :]
    sine = np.sin(latitude)
    cosine = np.cos(latitude)
    scalar = np.broadcast_to(cosine * np.cos(longitude), (grid.nlat, grid.nlon)).copy()
    radius = 2.5
    actual = sgj.gradient(_as_dtype(scalar, dtype), grid=grid, radius=radius)
    expected = (
        np.broadcast_to(-np.sin(longitude) / radius, scalar.shape),
        np.broadcast_to(-sine * np.cos(longitude) / radius, scalar.shape),
    )
    _assert_close(actual[0], expected[0])
    _assert_close(actual[1], expected[1])
    recovered = sgj.inverse_gradient(*actual, grid=grid, radius=radius)
    _assert_close(recovered, scalar)


@pytest.mark.parametrize("grid_name", ["gl_grid", "cc_grid"])
@pytest.mark.parametrize("dtype", [jnp.float64])
def test_spin_one_signs_and_atmospheric_operations(
    request: pytest.FixtureRequest,
    grid_name: str,
    dtype: object,
) -> None:
    grid = request.getfixturevalue(grid_name)
    radius = 2.5
    velocity_potential, streamfunction = scalar_potentials(grid)
    divergent_u, divergent_v = pure_gradient_values(grid, radius)
    rotational_u, rotational_v = pure_rotational_values(grid, radius)
    eastward, northward = vector_values(grid, radius)
    velocity_potential = np.asarray(velocity_potential)
    streamfunction = np.asarray(streamfunction)
    latitude = np.deg2rad(grid.latitude)[:, None]
    longitude = np.deg2rad(grid.longitude)[None, :]
    sine = np.sin(latitude)
    cosine = np.cos(latitude)
    lap_velocity_potential = (
        -2.0 * 1.4 * cosine * np.cos(longitude)
        - 6.0 * 0.6 * cosine**2 * np.cos(2.0 * longitude)
    ) / radius**2
    lap_streamfunction = (
        -2.0 * 1.2 * cosine * np.sin(longitude)
        - 6.0 * 0.8 * cosine**2 * np.sin(2.0 * longitude)
    ) / radius**2
    shape = (grid.nlat, grid.nlon)
    lap_velocity_potential = np.broadcast_to(lap_velocity_potential, shape)
    lap_streamfunction = np.broadcast_to(lap_streamfunction, shape)

    pure_divergent = sgj.kinematics(
        _as_dtype(divergent_u, dtype),
        _as_dtype(divergent_v, dtype),
        grid=grid,
        radius=radius,
    )
    _assert_close(pure_divergent[0], np.zeros(shape))
    _assert_close(pure_divergent[1], lap_velocity_potential)
    pure_rotational = sgj.kinematics(
        _as_dtype(rotational_u, dtype),
        _as_dtype(rotational_v, dtype),
        grid=grid,
        radius=radius,
    )
    _assert_close(pure_rotational[0], lap_streamfunction)
    _assert_close(pure_rotational[1], np.zeros(shape))

    actual_gradient = sgj.gradient(
        _as_dtype(velocity_potential, dtype),
        grid=grid,
        radius=radius,
    )
    _assert_close(actual_gradient[0], divergent_u)
    _assert_close(actual_gradient[1], divergent_v)
    recovered_velocity_potential = sgj.inverse_gradient(
        _as_dtype(divergent_u, dtype),
        _as_dtype(divergent_v, dtype),
        grid=grid,
        radius=radius,
    )
    _assert_close(recovered_velocity_potential, velocity_potential)

    projected_rotational = sgj.inverse_gradient(
        _as_dtype(rotational_u, dtype),
        _as_dtype(rotational_v, dtype),
        grid=grid,
        radius=radius,
    )
    _assert_close(projected_rotational, np.zeros(shape))

    projected_mixed = sgj.inverse_gradient(
        _as_dtype(divergent_u + rotational_u, dtype),
        _as_dtype(divergent_v + rotational_v, dtype),
        grid=grid,
        radius=radius,
    )
    _assert_close(projected_mixed, velocity_potential)

    u = _as_dtype(eastward, dtype)
    v = _as_dtype(northward, dtype)
    vo, div = sgj.kinematics(u, v, grid=grid, radius=radius)
    _assert_close(vo, lap_streamfunction)
    _assert_close(div, lap_velocity_potential)

    actual_potentials = sgj.potentials(u, v, grid=grid, radius=radius)
    _assert_close(actual_potentials[0], streamfunction)
    _assert_close(actual_potentials[1], velocity_potential)
    _assert_close(
        sgj.streamfunction(u, v, grid=grid, radius=radius),
        streamfunction,
    )
    _assert_close(
        sgj.velocity_potential(u, v, grid=grid, radius=radius),
        velocity_potential,
    )

    actual_helmholtz = sgj.helmholtz(u, v, grid=grid, radius=radius)
    for actual, expected in zip(
        actual_helmholtz,
        (divergent_u, divergent_v, rotational_u, rotational_v),
        strict=True,
    ):
        _assert_close(actual, expected)

    # The vector Laplacian applies the scalar eigenvalue to each potential.
    vector_laplacian_u = (
        2.0 * 1.4 * np.sin(longitude)
        + 12.0 * 0.6 * cosine * np.sin(2.0 * longitude)
        - 2.0 * 1.2 * sine * np.sin(longitude)
        - 12.0 * 0.8 * sine * cosine * np.sin(2.0 * longitude)
    ) / radius**3
    vector_laplacian_v = (
        2.0 * 1.4 * sine * np.cos(longitude)
        + 12.0 * 0.6 * sine * cosine * np.cos(2.0 * longitude)
        - 2.0 * 1.2 * np.cos(longitude)
        - 12.0 * 0.8 * cosine * np.cos(2.0 * longitude)
    ) / radius**3
    _assert_close(
        sgj.vector_laplacian(u, v, grid=grid, radius=radius)[0],
        np.broadcast_to(vector_laplacian_u, shape),
    )
    _assert_close(
        sgj.vector_laplacian(u, v, grid=grid, radius=radius)[1],
        np.broadcast_to(vector_laplacian_v, shape),
    )
    inverse_vector = sgj.inverse_vector_laplacian(
        *sgj.vector_laplacian(u, v, grid=grid, radius=radius),
        grid=grid,
        radius=radius,
    )
    _assert_close(inverse_vector[0], eastward)
    _assert_close(inverse_vector[1], northward)

    for source, field in (
        ("vorticity", lap_streamfunction),
        ("streamfunction", streamfunction),
    ):
        actual = sgj.rotational_wind(
            _as_dtype(field, dtype),
            grid=grid,
            source=source,
            radius=radius,
        )
        _assert_close(actual[0], rotational_u)
        _assert_close(actual[1], rotational_v)
    for source, field in (
        ("divergence", lap_velocity_potential),
        ("velocity_potential", velocity_potential),
    ):
        actual = sgj.divergent_wind(
            _as_dtype(field, dtype),
            grid=grid,
            source=source,
            radius=radius,
        )
        _assert_close(actual[0], divergent_u)
        _assert_close(actual[1], divergent_v)

    for source, first, second in (
        (
            "vorticity_divergence",
            lap_streamfunction,
            lap_velocity_potential,
        ),
        ("potentials", streamfunction, velocity_potential),
    ):
        actual = sgj.wind(
            _as_dtype(first, dtype),
            _as_dtype(second, dtype),
            grid=grid,
            source=source,
            radius=radius,
        )
        _assert_close(actual[0], eastward)
        _assert_close(actual[1], northward)
