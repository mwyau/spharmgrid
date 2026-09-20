# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Grid layout and leading-dimension behavior for S2FFT."""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
import numpy as np
import pytest

import spharmgrid as sg
import spharmgrid.jax as sgj
from tests.optional.jax._fields import as_xarray, scalar_values, vector_values


@pytest.mark.parametrize("kind", ["gl", "cc"])
@pytest.mark.parametrize("latitude_order", ["ascending", "descending"])
def test_exact_gl_and_mwss_shapes_accept_both_latitude_orders(
    kind: Literal["gl", "cc"],
    latitude_order: Literal["ascending", "descending"],
) -> None:
    from tests.optional.jax._fields import make_grid

    nlat = 8 if kind == "gl" else 9
    grid = make_grid(kind, nlat=nlat, latitude_order=latitude_order, lon0=37.0)
    values = scalar_values(grid)
    result = sgj.filter(jnp.asarray(values, dtype=jnp.float64), grid=grid)

    assert result.shape == values.shape
    np.testing.assert_allclose(result, values, rtol=2.0e-10, atol=2.0e-11)


@pytest.mark.parametrize("kind", ["gl", "cc"])
@pytest.mark.parametrize("lon0", [-180.0, 37.0])
def test_cyclic_longitude_order_and_nonzero_origin_are_restored(
    kind: Literal["gl", "cc"],
    lon0: float,
) -> None:
    from spharmgrid import Grid
    from tests.optional.jax._fields import make_grid

    base = make_grid(kind, nlat=8 if kind == "gl" else 9, lon0=lon0)
    shift = 3
    grid = Grid(base.kind, base.latitude, np.roll(base.longitude, shift))
    values = scalar_values(grid)
    field = jnp.asarray(values, dtype=jnp.float64)

    filtered = sgj.filter(field, "T4", grid=grid)
    expected = sgj.regrid(
        field,
        grid,
        "T4",
        source_grid=grid,
    )
    np.testing.assert_allclose(filtered, expected, rtol=2.0e-10, atol=2.0e-11)
    np.testing.assert_allclose(
        filtered,
        as_xarray(values, grid).values,
        rtol=2.0e-10,
        atol=2.0e-11,
    )


def test_scalar_and_vector_leading_dimensions_are_preserved(
    gl_grid: sg.Grid,
) -> None:
    scalar = jnp.asarray(scalar_values(gl_grid), dtype=jnp.float64)
    eastward, northward = vector_values(gl_grid)
    eastward = jnp.asarray(eastward, dtype=jnp.float64)
    northward = jnp.asarray(northward, dtype=jnp.float64)
    scalar_batch = jnp.stack((scalar, 2.0 * scalar), axis=0)
    eastward_batch = jnp.stack((eastward, 2.0 * eastward), axis=0)
    northward_batch = jnp.stack((northward, 2.0 * northward), axis=0)

    filtered = sgj.filter(scalar_batch, "T4", grid=gl_grid)
    output_u, output_v = sgj.regrid_vector(
        eastward_batch,
        northward_batch,
        gl_grid,
        "T4",
        source_grid=gl_grid,
    )

    assert filtered.shape == scalar_batch.shape
    assert output_u.shape == eastward_batch.shape
    assert output_v.shape == northward_batch.shape
    np.testing.assert_allclose(
        filtered[1], 2.0 * filtered[0], rtol=2.0e-10, atol=2.0e-11
    )
    np.testing.assert_allclose(
        output_u[1], 2.0 * output_u[0], rtol=2.0e-10, atol=2.0e-11
    )
    np.testing.assert_allclose(
        output_v[1], 2.0 * output_v[0], rtol=2.0e-10, atol=2.0e-11
    )
