# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spectral coefficient resizing across the supported GL and MWSS grids."""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
import numpy as np
import pytest
from jax import config

import spharmgrid.jax as sgj
from spharmgrid import Grid
from tests.optional.jax._fields import make_grid, scalar_values, vector_values


def _assert_close(actual: object, expected: np.ndarray) -> None:
    if not config.read("jax_enable_x64"):
        pytest.skip("float64 validation requires JAX x64 enabled")
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=2.0e-10, atol=2.0e-11)


@pytest.mark.parametrize(
    ("source_name", "target_name"),
    [
        ("gl_grid", "gl_target_grid"),
        ("gl_grid", "cc_target_grid"),
        ("cc_grid", "gl_target_grid"),
        ("cc_grid", "cc_target_grid"),
    ],
)
@pytest.mark.parametrize("dtype", [jnp.float64])
def test_scalar_regrid_resizes_s2fft_coefficients(
    request: pytest.FixtureRequest,
    source_name: str,
    target_name: str,
    dtype: object,
) -> None:
    source_grid = request.getfixturevalue(source_name)
    target_grid = request.getfixturevalue(target_name)
    source_values = scalar_values(source_grid)
    expected = scalar_values(target_grid)
    result = sgj.regrid(
        jnp.asarray(source_values, dtype=dtype),
        target_grid,
        source_grid=source_grid,
    )
    _assert_close(result, expected)


@pytest.mark.parametrize(
    ("source_name", "target_name"),
    [
        ("gl_grid", "gl_target_grid"),
        ("gl_grid", "cc_target_grid"),
        ("cc_grid", "gl_target_grid"),
        ("cc_grid", "cc_target_grid"),
    ],
)
@pytest.mark.parametrize("dtype", [jnp.float64])
def test_vector_regrid_resizes_centered_m_modes(
    request: pytest.FixtureRequest,
    source_name: str,
    target_name: str,
    dtype: object,
) -> None:
    source_grid = request.getfixturevalue(source_name)
    target_grid = request.getfixturevalue(target_name)
    source_u, source_v = vector_values(source_grid)
    expected_u, expected_v = vector_values(target_grid)
    actual_u, actual_v = sgj.regrid_vector(
        jnp.asarray(source_u, dtype=dtype),
        jnp.asarray(source_v, dtype=dtype),
        target_grid,
        "T4",
        source_grid=source_grid,
    )
    _assert_close(actual_u, expected_u)
    _assert_close(actual_v, expected_v)


def _grid_for_bandlimit(kind: Literal["gl", "cc"], bandlimit: int) -> Grid:
    nlat = bandlimit if kind == "gl" else bandlimit + 1
    return make_grid(kind, nlat=nlat, latitude_order="ascending", lon0=19.0)


def _sectoral_values(grid: Grid, degree: int, amplitude: float = 1.0) -> np.ndarray:
    latitude = np.deg2rad(grid.latitude)[:, None]
    longitude = np.deg2rad(grid.longitude)[None, :]
    return amplitude * np.cos(latitude) ** degree * np.cos(degree * longitude)


def _sectoral_gradient(
    grid: Grid, degree: int, amplitude: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    latitude = np.deg2rad(grid.latitude)[:, None]
    longitude = np.deg2rad(grid.longitude)[None, :]
    cosine = np.cos(latitude)
    sine = np.sin(latitude)
    eastward = -amplitude * degree * cosine ** (degree - 1) * np.sin(degree * longitude)
    northward = (
        -amplitude * degree * sine * cosine ** (degree - 1) * np.cos(degree * longitude)
    )
    shape = (grid.nlat, grid.nlon)
    return (
        np.broadcast_to(eastward, shape).copy(),
        np.broadcast_to(northward, shape).copy(),
    )


@pytest.mark.parametrize("kind", ["gl", "cc"])
@pytest.mark.parametrize("dtype", [jnp.float64])
def test_regrid_preserves_near_edge_sectoral_mode(
    kind: Literal["gl", "cc"],
    dtype: object,
) -> None:
    source_grid = _grid_for_bandlimit(kind, 12)
    target_grid = _grid_for_bandlimit(kind, 16)
    values = _sectoral_values(source_grid, 11)
    expected = _sectoral_values(target_grid, 11)

    actual = sgj.regrid(
        jnp.asarray(values, dtype=dtype),
        target_grid,
        source_grid=source_grid,
    )

    _assert_close(actual, expected)


@pytest.mark.parametrize("kind", ["gl", "cc"])
@pytest.mark.parametrize("dtype", [jnp.float64])
def test_regrid_removes_above_target_sectoral_mode(
    kind: Literal["gl", "cc"],
    dtype: object,
) -> None:
    source_grid = _grid_for_bandlimit(kind, 16)
    target_grid = _grid_for_bandlimit(kind, 12)
    values = _sectoral_values(source_grid, 3, amplitude=0.6) + _sectoral_values(
        source_grid, 15, amplitude=0.8
    )
    expected = _sectoral_values(target_grid, 3, amplitude=0.6)

    actual = sgj.regrid(
        jnp.asarray(values, dtype=dtype),
        target_grid,
        source_grid=source_grid,
    )

    _assert_close(actual, expected)


@pytest.mark.parametrize("kind", ["gl", "cc"])
@pytest.mark.parametrize("dtype", [jnp.float64])
def test_vector_regrid_preserves_near_edge_sectoral_mode(
    kind: Literal["gl", "cc"],
    dtype: object,
) -> None:
    source_grid = _grid_for_bandlimit(kind, 12)
    target_grid = _grid_for_bandlimit(kind, 16)
    source_u, source_v = _sectoral_gradient(source_grid, 11)
    expected_u, expected_v = _sectoral_gradient(target_grid, 11)

    actual_u, actual_v = sgj.regrid_vector(
        jnp.asarray(source_u, dtype=dtype),
        jnp.asarray(source_v, dtype=dtype),
        target_grid,
        source_grid=source_grid,
    )

    _assert_close(actual_u, expected_u)
    _assert_close(actual_v, expected_v)
