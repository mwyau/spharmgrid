# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""JAX precision and process-configuration requirements."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax import config

import spharmgrid as sg
import spharmgrid.jax as sgj
from tests.optional.jax._fields import scalar_values


def test_x64_disabled_rejects_single_precision_execution(gl_grid: sg.Grid) -> None:
    if config.read("jax_enable_x64"):
        pytest.skip("this check targets a process with JAX x64 disabled")

    field = jnp.asarray(scalar_values(gl_grid), dtype=jnp.float32)
    with pytest.raises(RuntimeError, match="requires JAX x64 mode"):
        sgj.filter(field, grid=gl_grid)


def test_float32_scalar_input_is_promoted_to_float64(gl_grid: sg.Grid) -> None:
    if not config.read("jax_enable_x64"):
        pytest.skip("float32 promotion requires JAX x64")

    field = jnp.asarray(scalar_values(gl_grid), dtype=jnp.float32)
    actual = sgj.filter(field, grid=gl_grid)
    expected = sgj.filter(field.astype(jnp.float64), grid=gl_grid)

    assert actual.dtype == jnp.float64
    np.testing.assert_allclose(actual, expected, rtol=1.0e-13, atol=1.0e-13)


def test_float32_vector_input_is_promoted_to_float64(gl_grid: sg.Grid) -> None:
    if not config.read("jax_enable_x64"):
        pytest.skip("float32 promotion requires JAX x64")

    u = jnp.asarray(scalar_values(gl_grid), dtype=jnp.float32)
    v = jnp.roll(u, 1, axis=-1)
    actual = sgj.regrid_vector(u, v, gl_grid, source_grid=gl_grid)
    expected = sgj.regrid_vector(
        u.astype(jnp.float64),
        v.astype(jnp.float64),
        gl_grid,
        source_grid=gl_grid,
    )

    for actual_component, expected_component in zip(actual, expected, strict=True):
        assert actual_component.dtype == jnp.float64
        np.testing.assert_allclose(
            actual_component,
            expected_component,
            rtol=1.0e-13,
            atol=1.0e-13,
        )


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.complex64])
def test_unsupported_input_dtypes_are_rejected_with_x64_enabled(
    gl_grid: sg.Grid, dtype: object
) -> None:
    if not config.read("jax_enable_x64"):
        pytest.skip("dtype checks require JAX x64")

    field = jnp.ones((gl_grid.nlat, gl_grid.nlon), dtype=dtype)
    with pytest.raises(TypeError, match="must use float32 or float64"):
        sgj.filter(field, grid=gl_grid)
