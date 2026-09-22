# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Analytic scalar, vector, and atmospheric-operator tests for JAX."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from jax.typing import DTypeLike

import spharmgrid.jax as sgj
from tests.optional.jax._fields import (
    pure_gradient_values,
    pure_rotational_values,
    scalar_potentials,
    vector_values,
)

pytestmark = pytest.mark.jax_x64


def _as_dtype(values: np.ndarray, dtype: DTypeLike) -> Array:
    return jnp.asarray(values, dtype=dtype)


def _assert_close(actual: object, expected: object, *, atol: float) -> None:
    actual_values = np.asarray(actual)
    expected_values = np.broadcast_to(np.asarray(expected), actual_values.shape)
    np.testing.assert_allclose(
        actual_values,
        expected_values,
        rtol=0.0,
        atol=atol,
    )


@pytest.mark.parametrize("grid_name", ["gl_grid", "cc_grid"])
@pytest.mark.parametrize("dtype", [jnp.float64])
def test_scalar_harmonic_identities(
    request: pytest.FixtureRequest,
    grid_name: str,
    dtype: DTypeLike,
) -> None:
    # CI measured a 3.63e-13 maximum absolute Laplacian error on GL.
    atol = 5.0e-13
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
    _assert_close(low, 1.25 + 0.5 * degree_one, atol=atol)
    _assert_close(
        band,
        0.75 * degree_two_zonal + 0.2 * degree_two_sectoral,
        atol=atol,
    )
    _assert_close(explicit_band, band, atol=atol)
    _assert_close(tapered, 0.2 * degree_two_sectoral, atol=atol)

    expected_laplacian = (
        -1.0 * degree_one - 4.5 * degree_two_zonal - 1.2 * degree_two_sectoral
    ) / radius**2
    expected_inverse = radius**2 * (
        -0.25 * degree_one
        - 0.125 * degree_two_zonal
        - (0.2 / 6.0) * degree_two_sectoral
    )
    _assert_close(
        sgj.laplacian(array, grid=grid, radius=radius),
        expected_laplacian,
        atol=atol,
    )
    _assert_close(
        sgj.inverse_laplacian(array, grid=grid, radius=radius),
        expected_inverse,
        atol=atol,
    )


@pytest.mark.parametrize("grid_name", ["gl_grid", "cc_grid"])
@pytest.mark.parametrize("dtype", [jnp.float64])
def test_gradient_and_inverse_gradient_use_geographic_components(
    request: pytest.FixtureRequest,
    grid_name: str,
    dtype: DTypeLike,
) -> None:
    atol = 2.0e-13
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
    _assert_close(actual[0], expected[0], atol=atol)
    _assert_close(actual[1], expected[1], atol=atol)
    recovered = sgj.inverse_gradient(*actual, grid=grid, radius=radius)
    _assert_close(recovered, scalar, atol=atol)


@pytest.mark.parametrize("grid_name", ["gl_grid", "cc_grid"])
@pytest.mark.parametrize("dtype", [jnp.float64])
def test_spin_one_signs_and_atmospheric_operations(
    request: pytest.FixtureRequest,
    grid_name: str,
    dtype: DTypeLike,
) -> None:
    atol = 3.0e-13
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
    _assert_close(pure_divergent[0], np.zeros(shape), atol=atol)
    _assert_close(pure_divergent[1], lap_velocity_potential, atol=atol)
    pure_rotational = sgj.kinematics(
        _as_dtype(rotational_u, dtype),
        _as_dtype(rotational_v, dtype),
        grid=grid,
        radius=radius,
    )
    _assert_close(pure_rotational[0], lap_streamfunction, atol=atol)
    _assert_close(pure_rotational[1], np.zeros(shape), atol=atol)

    actual_gradient = sgj.gradient(
        _as_dtype(velocity_potential, dtype),
        grid=grid,
        radius=radius,
    )
    _assert_close(actual_gradient[0], divergent_u, atol=atol)
    _assert_close(actual_gradient[1], divergent_v, atol=atol)
    recovered_velocity_potential = sgj.inverse_gradient(
        _as_dtype(divergent_u, dtype),
        _as_dtype(divergent_v, dtype),
        grid=grid,
        radius=radius,
    )
    _assert_close(recovered_velocity_potential, velocity_potential, atol=atol)

    projected_rotational = sgj.inverse_gradient(
        _as_dtype(rotational_u, dtype),
        _as_dtype(rotational_v, dtype),
        grid=grid,
        radius=radius,
    )
    _assert_close(projected_rotational, np.zeros(shape), atol=atol)

    projected_mixed = sgj.inverse_gradient(
        _as_dtype(divergent_u + rotational_u, dtype),
        _as_dtype(divergent_v + rotational_v, dtype),
        grid=grid,
        radius=radius,
    )
    _assert_close(projected_mixed, velocity_potential, atol=atol)

    u = _as_dtype(eastward, dtype)
    v = _as_dtype(northward, dtype)
    vo, div = sgj.kinematics(u, v, grid=grid, radius=radius)
    _assert_close(vo, lap_streamfunction, atol=atol)
    _assert_close(div, lap_velocity_potential, atol=atol)

    actual_potentials = sgj.potentials(u, v, grid=grid, radius=radius)
    _assert_close(actual_potentials[0], streamfunction, atol=atol)
    _assert_close(actual_potentials[1], velocity_potential, atol=atol)
    _assert_close(
        sgj.streamfunction(u, v, grid=grid, radius=radius),
        streamfunction,
        atol=atol,
    )
    _assert_close(
        sgj.velocity_potential(u, v, grid=grid, radius=radius),
        velocity_potential,
        atol=atol,
    )

    actual_helmholtz = sgj.helmholtz(u, v, grid=grid, radius=radius)
    for actual, expected in zip(
        actual_helmholtz,
        (divergent_u, divergent_v, rotational_u, rotational_v),
        strict=True,
    ):
        _assert_close(actual, expected, atol=atol)

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
        atol=atol,
    )
    _assert_close(
        sgj.vector_laplacian(u, v, grid=grid, radius=radius)[1],
        np.broadcast_to(vector_laplacian_v, shape),
        atol=atol,
    )
    inverse_vector = sgj.inverse_vector_laplacian(
        *sgj.vector_laplacian(u, v, grid=grid, radius=radius),
        grid=grid,
        radius=radius,
    )
    _assert_close(inverse_vector[0], eastward, atol=atol)
    _assert_close(inverse_vector[1], northward, atol=atol)

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
        _assert_close(actual[0], rotational_u, atol=atol)
        _assert_close(actual[1], rotational_v, atol=atol)
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
        _assert_close(actual[0], divergent_u, atol=atol)
        _assert_close(actual[1], divergent_v, atol=atol)

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
        _assert_close(actual[0], eastward, atol=atol)
        _assert_close(actual[1], northward, atol=atol)
