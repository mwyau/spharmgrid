# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""JAX precision and process-configuration requirements."""

from __future__ import annotations

import jax.numpy as jnp
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


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.complex64])
def test_single_precision_inputs_are_rejected_with_x64_enabled(
    gl_grid: sg.Grid, dtype: object
) -> None:
    if not config.read("jax_enable_x64"):
        pytest.skip("single-precision dtype checks require JAX x64")

    field = jnp.ones((gl_grid.nlat, gl_grid.nlon), dtype=dtype)
    with pytest.raises(TypeError, match="must use float64"):
        sgj.filter(field, grid=gl_grid)
