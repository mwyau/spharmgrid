# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scalar differential-operator tests."""

from __future__ import annotations

import numpy as np
import pytest

import spharmgrid as sg
from tests.conftest import (
    SupportedGridKind,
    degree_one_field,
    scalar_field,
    supported_grid,
)


@pytest.mark.parametrize("kind", ["cc", "gl", "f1"])
def test_gradient_laplacian_and_inverse_laplacian_of_degree_one(
    kind: SupportedGridKind,
) -> None:
    grid = supported_grid(kind)
    field = degree_one_field(grid)
    latitude = np.deg2rad(grid.latitude)[:, None]
    radius = sg.EARTH_RADIUS_M

    gradient = sg.gradient(field)
    laplacian = sg.laplacian(field)
    inverse = sg.inverse_laplacian(field)

    expected_north = np.cos(latitude) / radius * np.ones((1, grid.nlon))
    np.testing.assert_allclose(gradient.gradient_eastward, 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        gradient.gradient_northward, expected_north, rtol=0.0, atol=1.0e-20
    )
    np.testing.assert_allclose(
        laplacian, -2.0 * field / radius**2, rtol=0.0, atol=5.0e-26
    )
    # The inverse multiplier is O(R**2), so coefficient roundoff is amplified
    # to a few 1e-3 in physical space across the supported platforms.
    np.testing.assert_allclose(
        inverse,
        -(radius**2) * field / 2.0,
        rtol=0.0,
        atol=1.0e-2,
    )


@pytest.mark.parametrize("kind", ["cc", "gl", "f1"])
def test_inverse_laplacian_sets_degree_zero_to_zero(kind: SupportedGridKind) -> None:
    grid = supported_grid(kind)
    field = degree_one_field(grid) + 7.0

    inverse = sg.inverse_laplacian(field)
    restored = sg.laplacian(inverse)

    np.testing.assert_allclose(
        inverse,
        -(sg.EARTH_RADIUS_M**2) * degree_one_field(grid) / 2.0,
        rtol=0.0,
        # Roundoff in the degree-one coefficient is amplified by R**2 after
        # analyzing a field that also contains a large constant.
        atol=2.0e-1,
    )
    np.testing.assert_allclose(restored, degree_one_field(grid), rtol=0.0, atol=1.0e-12)


@pytest.mark.parametrize("kind", ["cc", "gl", "f1"])
def test_scalar_laplacian_degree_two_harmonics(kind: SupportedGridKind) -> None:
    """Check degree-one and degree-two eigenvalues in physical space."""
    grid = supported_grid(kind)
    source = scalar_field(grid)
    latitude = np.deg2rad(grid.latitude)[:, None]
    longitude = np.deg2rad(grid.longitude)[None, :]
    sine = np.sin(latitude)

    degree_one = 0.5 * sine
    sectoral = np.cos(latitude) ** 2 * np.cos(2.0 * longitude)
    degree_two = 0.75 * (3.0 * sine**2 - 1.0) / 2.0 + 0.2 * sectoral
    radius = sg.EARTH_RADIUS_M

    expected_laplacian = -(2.0 * degree_one + 6.0 * degree_two) / radius**2
    expected_inverse = -(radius**2) * (degree_one / 2.0 + degree_two / 6.0)
    np.testing.assert_allclose(
        sg.laplacian(source), expected_laplacian, rtol=0.0, atol=1.0e-25
    )
    np.testing.assert_allclose(
        sg.inverse_laplacian(source), expected_inverse, rtol=0.0, atol=1.0
    )


def test_scalar_laplacian_units_simplify_operator_chains() -> None:
    field = degree_one_field(supported_grid("cc"))
    field.attrs["units"] = "K"

    laplacian = sg.laplacian(field)
    restored = sg.inverse_laplacian(laplacian)
    inverse = sg.inverse_laplacian(field)
    inverse_restored = sg.laplacian(inverse)

    assert laplacian.attrs["units"] == "K m-2"
    assert restored.attrs["units"] == "K"
    assert inverse.attrs["units"] == "K m2"
    assert inverse_restored.attrs["units"] == "K"
