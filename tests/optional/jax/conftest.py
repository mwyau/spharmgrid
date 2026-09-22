# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixtures and optional dependency loading for the JAX tests."""

from __future__ import annotations

import importlib

import pytest

import spharmgrid as sg

jax = pytest.importorskip("jax")
pytest.importorskip("s2fft")
jnp = importlib.import_module("jax.numpy")
sgj = importlib.import_module("spharmgrid.jax")


@pytest.fixture(autouse=True)
def _skip_jax_x64_disabled(request: pytest.FixtureRequest) -> None:
    if request.node.get_closest_marker("jax_x64") is not None and not jax.config.read(
        "jax_enable_x64"
    ):
        pytest.skip("JAX x64 mode is required; run with JAX_ENABLE_X64=1")


@pytest.fixture(scope="module")
def gl_grid() -> sg.Grid:
    return sg.gaussian_grid(8, 15, latitude_order="descending", lon0=37.0)


@pytest.fixture(scope="module")
def cc_grid() -> sg.Grid:
    return sg.clenshaw_curtis_grid(
        9,
        16,
        latitude_order="ascending",
        lon0=-75.0,
    )


@pytest.fixture(scope="module")
def gl_target_grid() -> sg.Grid:
    return sg.gaussian_grid(6, 11, latitude_order="ascending", lon0=123.0)


@pytest.fixture(scope="module")
def cc_target_grid() -> sg.Grid:
    return sg.clenshaw_curtis_grid(
        7,
        12,
        latitude_order="descending",
        lon0=123.0,
    )


__all__ = ["jax", "jnp", "sg", "sgj"]
