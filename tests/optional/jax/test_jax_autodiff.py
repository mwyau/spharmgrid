# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""JAX transformation and automatic-differentiation checks."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax import checking_leaks, config, grad, jit, vmap

import spharmgrid as sg
import spharmgrid.jax as sgj
from spharmgrid.jax._backend import _precomputes
from tests.optional.jax._fields import scalar_values, vector_values


def _require_x64() -> None:
    if not config.read("jax_enable_x64"):
        pytest.skip("finite-difference reference requires JAX x64")


def test_cold_precompute_cache_does_not_leak_under_jit(
    gl_grid: sg.Grid,
) -> None:
    field = jnp.asarray(scalar_values(gl_grid), dtype=jnp.float64)
    _precomputes.cache_clear()

    with checking_leaks():
        filtered = jit(lambda values: sgj.filter(values, "T4", grid=gl_grid))(field)
        filtered.block_until_ready()

    assert filtered.shape == field.shape


def test_scalar_filter_and_laplacian_support_jit_vmap_and_grad(
    gl_grid: sg.Grid,
) -> None:
    field = jnp.asarray(scalar_values(gl_grid), dtype=jnp.float64)
    batch = jnp.stack((field, 1.5 * field), axis=0)

    compiled_filter = jit(lambda values: sgj.filter(values, "T4", grid=gl_grid))
    compiled_laplacian = jit(lambda values: sgj.laplacian(values, grid=gl_grid))
    filtered = compiled_filter(field)
    laplacian = compiled_laplacian(field)
    filtered_batch = vmap(lambda values: sgj.filter(values, "T4", grid=gl_grid))(batch)
    filter_gradient = grad(
        lambda values: jnp.sum(sgj.filter(values, "T4", grid=gl_grid) ** 2)
    )(field)
    laplacian_gradient = grad(
        lambda values: jnp.sum(sgj.laplacian(values, grid=gl_grid) ** 2)
    )(field)

    assert filtered.shape == field.shape
    assert laplacian.shape == field.shape
    assert filtered_batch.shape == batch.shape
    assert filter_gradient.shape == field.shape
    assert laplacian_gradient.shape == field.shape
    assert bool(jnp.isfinite(filter_gradient).all())
    assert bool(jnp.isfinite(laplacian_gradient).all())


def test_vector_kinematics_and_wind_support_jit_vmap_and_grad(
    cc_grid: sg.Grid,
) -> None:
    eastward_values, northward_values = vector_values(cc_grid)
    eastward = jnp.asarray(eastward_values, dtype=jnp.float64)
    northward = jnp.asarray(northward_values, dtype=jnp.float64)
    eastward_batch = jnp.stack((eastward, 1.5 * eastward), axis=0)
    northward_batch = jnp.stack((northward, 1.5 * northward), axis=0)
    weights = jnp.asarray(
        np.linspace(0.25, 1.0, cc_grid.nlat * cc_grid.nlon), dtype=jnp.float64
    ).reshape(cc_grid.nlat, cc_grid.nlon)

    compiled_kinematics = jit(lambda u, v: sgj.kinematics(u, v, grid=cc_grid))
    compiled_wind = jit(
        lambda first, second: sgj.wind(
            first,
            second,
            grid=cc_grid,
            source="vorticity_divergence",
        )
    )
    vorticity, divergence = compiled_kinematics(eastward, northward)
    vector_batch = vmap(lambda u, v: sgj.kinematics(u, v, grid=cc_grid))(
        eastward_batch, northward_batch
    )
    reconstructed_u, reconstructed_v = compiled_wind(vorticity, divergence)
    kinematics_gradient = grad(
        lambda u: jnp.sum(sgj.kinematics(u, northward, grid=cc_grid)[0] * weights)
    )(eastward)
    wind_gradient = grad(
        lambda first: jnp.sum(
            sgj.wind(
                first,
                divergence,
                grid=cc_grid,
                source="vorticity_divergence",
            )[0]
            * weights
        )
    )(vorticity)

    assert vorticity.shape == eastward.shape
    assert divergence.shape == eastward.shape
    assert vector_batch[0].shape == eastward_batch.shape
    assert vector_batch[1].shape == northward_batch.shape
    assert reconstructed_u.shape == eastward.shape
    assert reconstructed_v.shape == northward.shape
    assert kinematics_gradient.shape == eastward.shape
    assert wind_gradient.shape == vorticity.shape
    assert bool(jnp.isfinite(kinematics_gradient).all())
    assert bool(jnp.isfinite(wind_gradient).all())


def test_scalar_filter_directional_derivative_matches_finite_difference(
    gl_grid: sg.Grid,
) -> None:
    _require_x64()
    field = jnp.asarray(scalar_values(gl_grid), dtype=jnp.float64)
    direction = jnp.asarray(
        np.sin(np.linspace(0.2, 2.8, gl_grid.nlat * gl_grid.nlon)).reshape(
            gl_grid.nlat, gl_grid.nlon
        ),
        dtype=jnp.float64,
    )
    weights = jnp.asarray(
        np.linspace(0.35, 1.25, gl_grid.nlat * gl_grid.nlon).reshape(
            gl_grid.nlat, gl_grid.nlon
        ),
        dtype=jnp.float64,
    )

    def loss(values: jnp.ndarray) -> jnp.ndarray:
        filtered = sgj.filter(values, "T4", grid=gl_grid)
        return jnp.sum(weights * filtered**2)

    derivative = jnp.sum(grad(loss)(field) * direction)
    epsilon = 1.0e-4
    finite_difference = (
        loss(field + epsilon * direction) - loss(field - epsilon * direction)
    ) / (2.0 * epsilon)

    np.testing.assert_allclose(
        derivative,
        finite_difference,
        rtol=2.0e-10,
        atol=2.0e-11,
    )


def test_vector_kinematics_directional_derivative_matches_finite_difference(
    cc_grid: sg.Grid,
) -> None:
    _require_x64()
    eastward_values, northward_values = vector_values(cc_grid)
    eastward = jnp.asarray(eastward_values, dtype=jnp.float64)
    northward = jnp.asarray(northward_values, dtype=jnp.float64)
    eastward_direction = jnp.asarray(
        np.cos(np.linspace(0.1, 2.5, cc_grid.nlat * cc_grid.nlon)).reshape(
            cc_grid.nlat, cc_grid.nlon
        ),
        dtype=jnp.float64,
    )
    northward_direction = jnp.asarray(
        np.sin(np.linspace(0.4, 2.9, cc_grid.nlat * cc_grid.nlon)).reshape(
            cc_grid.nlat, cc_grid.nlon
        ),
        dtype=jnp.float64,
    )
    weights = jnp.asarray(
        np.linspace(0.4, 1.1, cc_grid.nlat * cc_grid.nlon).reshape(
            cc_grid.nlat, cc_grid.nlon
        ),
        dtype=jnp.float64,
    )

    def loss(first: jnp.ndarray, second: jnp.ndarray) -> jnp.ndarray:
        vorticity, divergence = sgj.kinematics(
            first,
            second,
            grid=cc_grid,
            radius=1.0,
        )
        return jnp.sum(weights * (vorticity**2 + 0.7 * divergence**2))

    gradient_u, gradient_v = grad(loss, argnums=(0, 1))(eastward, northward)
    derivative = jnp.sum(
        gradient_u * eastward_direction + gradient_v * northward_direction
    )
    epsilon = 1.0e-4
    finite_difference = (
        loss(
            eastward + epsilon * eastward_direction,
            northward + epsilon * northward_direction,
        )
        - loss(
            eastward - epsilon * eastward_direction,
            northward - epsilon * northward_direction,
        )
    ) / (2.0 * epsilon)

    np.testing.assert_allclose(
        derivative,
        finite_difference,
        rtol=2.0e-10,
        atol=2.0e-11,
    )
