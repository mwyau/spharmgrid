# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spectral specification, selection, and validation tests."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
import xarray as xr

import spharmgrid as sg
from spharmgrid._ducc import (
    alm_degrees,
    alm_orders,
    geometry_for,
    scalar_synthesis,
)
from spharmgrid._transform import TransformSpec
from spharmgrid.grids import grid_layout
from spharmgrid.spectral import apply_spectral_selection, resolve_transform_spec
from tests.conftest import scalar_field, supported_grid


def _scalar_mode(
    grid: sg.Grid,
    degree: int,
    order: int,
    *,
    spec: TransformSpec,
) -> xr.DataArray:
    degrees = alm_degrees(spec.lmax, spec.mmax)
    index = np.flatnonzero(
        (degrees == degree) & (alm_orders(spec.lmax, spec.mmax) == order)
    )
    assert index.size == 1
    coefficients = np.zeros((1, degrees.size), dtype=np.complex128)
    coefficients[0, index[0]] = 1.0
    layout = grid_layout(grid)
    values = scalar_synthesis(
        coefficients,
        spec=spec,
        geometry=geometry_for(grid),
        ntheta=grid.nlat,
        nphi=grid.nlon,
        phi0=layout.phi0_radians,
        nthreads=0,
    )
    return xr.DataArray(
        values,
        dims=("lat", "lon"),
        coords={"lat": grid.latitude, "lon": grid.longitude},
    )


@pytest.mark.parametrize(
    ("notation", "expected"),
    [
        ("T42", TransformSpec(0, 42, 42, "triangular")),
        ("t6-42", TransformSpec(6, 42, 42, "triangular")),
        ("T6–42", TransformSpec(6, 42, 42, "triangular")),
    ],
)
def test_parse_spectral(notation: str, expected: TransformSpec) -> None:
    result = sg.parse_spectral(notation)

    assert result == expected


@pytest.mark.parametrize(
    ("notation", "expected"),
    [
        ("T42x10", TransformSpec(0, 42, 10, "trapezoidal")),
        ("t42X10", TransformSpec(0, 42, 10, "trapezoidal")),
        ("R42", TransformSpec(0, 84, 42, "rhomboidal")),
        ("r42", TransformSpec(0, 84, 42, "rhomboidal")),
    ],
)
def test_parse_nontriangular_spectral(
    notation: str,
    expected: TransformSpec,
) -> None:
    result = sg.parse_spectral(notation)

    assert result == expected


@pytest.mark.parametrize(
    "notation",
    ["", "T", "T42x", "T42-10", "T42x43", "R42x10", "Q42", "T5-42x10"],
)
def test_parse_spectral_rejects_malformed_shapes(notation: str) -> None:
    with pytest.raises(
        ValueError,
        match="spectral notation|spectral bounds|transform limits",
    ):
        sg.parse_spectral(notation)


def test_transform_spec_resolves_supported_shapes_and_full_domain() -> None:
    grid = sg.gaussian_grid(10, 20)

    assert resolve_transform_spec(grid, grid, sg.parse_spectral("T8")) == (
        TransformSpec(0, 8, 8, "triangular")
    )
    assert resolve_transform_spec(grid, grid, sg.parse_spectral("T8x3")) == (
        TransformSpec(0, 8, 3, "trapezoidal")
    )
    assert resolve_transform_spec(grid, grid, sg.parse_spectral("R4")) == (
        TransformSpec(0, 8, 4, "rhomboidal")
    )

    latitude_limited_by_grid = sg.gaussian_grid(8, 6)
    assert resolve_transform_spec(
        latitude_limited_by_grid, latitude_limited_by_grid, None
    ) == (TransformSpec(0, 7, 2, "trapezoidal"))


def test_nontriangular_capabilities_check_latitude_and_longitude_independently() -> (
    None
):
    low_latitude = sg.gaussian_grid(5, 40)
    high_bandwidth = sg.gaussian_grid(8, 40)
    low_longitude = sg.gaussian_grid(8, 6)

    with pytest.raises(ValueError, match="source latitude"):
        resolve_transform_spec(low_latitude, high_bandwidth, sg.parse_spectral("T5x2"))
    with pytest.raises(ValueError, match="target longitude"):
        resolve_transform_spec(high_bandwidth, low_longitude, sg.parse_spectral("T5x3"))


def test_alm_indices_follow_ducc_packed_order_and_rhomboidal_mask() -> None:
    degrees = alm_degrees(4, 2)
    orders = alm_orders(4, 2)
    assert list(zip(degrees.tolist(), orders.tolist(), strict=True)) == [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 1),
        (2, 2),
        (3, 2),
        (4, 2),
    ]

    spec = sg.parse_spectral("R2")
    coefficients = np.ones((2, degrees.size), dtype=np.complex128)
    selected = apply_spectral_selection(coefficients, spec, taper=None)
    expected = (degrees - orders <= spec.lmax - spec.mmax).astype(np.float64)
    np.testing.assert_allclose(selected, np.tile(expected, (2, 1)))

    tapered = apply_spectral_selection(coefficients, spec, taper=0.1)
    boundary = (degrees == 4) & (orders == 2)
    np.testing.assert_allclose(tapered[:, boundary], 0.1)
    np.testing.assert_allclose(tapered[:, (degrees == 3) & (orders == 0)], 0.0)


def test_trapezoidal_filter_retains_the_lmax_corner() -> None:
    grid = sg.clenshaw_curtis_grid(8, 12, latitude_order="descending")
    spec = TransformSpec(0, 4, 2, "trapezoidal")
    field = _scalar_mode(grid, 4, 2, spec=spec)

    result = sg.filter(field, "T4x2")

    np.testing.assert_allclose(result, field, rtol=0.0, atol=2.0e-13)


def test_rhomboidal_filter_removes_outside_modes_and_keeps_boundary_modes() -> None:
    grid = sg.clenshaw_curtis_grid(8, 12, latitude_order="descending")
    spec = TransformSpec(0, 4, 2, "trapezoidal")
    outside = _scalar_mode(grid, 3, 0, spec=spec)
    boundary = _scalar_mode(grid, 4, 2, spec=spec)

    outside_result = sg.filter(outside, "R2")
    boundary_result = sg.filter(boundary, "R2")
    tapered_boundary = sg.filter(boundary, "R2", taper=0.1)

    np.testing.assert_allclose(outside_result, 0.0, rtol=0.0, atol=2.0e-13)
    np.testing.assert_allclose(boundary_result, boundary, rtol=0.0, atol=2.0e-13)
    np.testing.assert_allclose(tapered_boundary, 0.1 * boundary, rtol=0.0, atol=2.0e-13)


def test_t5_42_remains_a_triangular_degree_band() -> None:
    grid = sg.gaussian_grid(44, 86, latitude_order="descending")
    spec = TransformSpec(0, 42, 42, "triangular")
    retained = _scalar_mode(grid, 6, 0, spec=spec)
    removed = _scalar_mode(grid, 3, 0, spec=spec)
    field = retained + removed

    parsed = sg.filter(field, "T5-42")
    explicit = sg.filter(field, lmin=5, lmax=42)

    np.testing.assert_allclose(parsed, retained, rtol=0.0, atol=3.0e-12)
    np.testing.assert_allclose(explicit, parsed, rtol=0.0, atol=3.0e-12)


@pytest.mark.parametrize("kind", ["cc", "gl"])
def test_hard_degree_selection_and_bandpass(kind: Literal["cc", "gl"]) -> None:
    grid = supported_grid(kind)
    latitude = np.sin(np.deg2rad(grid.latitude))[:, None]
    degree_zero = np.ones((grid.nlat, grid.nlon))
    degree_one = latitude * np.ones((1, grid.nlon))
    degree_two = (3.0 * latitude**2 - 1.0) / 2.0 * np.ones((1, grid.nlon))
    field = xr.DataArray(
        degree_zero + 2.0 * degree_one + 3.0 * degree_two,
        dims=("lat", "lon"),
        coords={"lat": grid.latitude, "lon": grid.longitude},
    )

    low = sg.filter(field, "T1")
    band = sg.filter(field, "T2-2")
    explicit_band = sg.filter(field, lmin=2, lmax=2)

    np.testing.assert_allclose(
        low, degree_zero + 2.0 * degree_one, rtol=0.0, atol=5.0e-15
    )
    np.testing.assert_allclose(band, 3.0 * degree_two, rtol=0.0, atol=1.0e-14)
    xr.testing.assert_identical(explicit_band, band)


@pytest.mark.parametrize("kind", ["cc", "gl"])
def test_t42_is_accepted_on_a_grid_that_represents_it(
    kind: Literal["cc", "gl"],
) -> None:
    grid = sg.clenshaw_curtis_grid(44, 85) if kind == "cc" else sg.gaussian_grid(43, 85)
    field = scalar_field(grid)

    result = sg.filter(field, "T42")

    np.testing.assert_allclose(result, field, rtol=0.0, atol=2.0e-13)


@pytest.mark.parametrize("kind", ["cc", "gl"])
def test_taper_response_at_upper_retained_degree(kind: Literal["cc", "gl"]) -> None:
    grid = supported_grid(kind)
    latitude = np.sin(np.deg2rad(grid.latitude))[:, None]
    degree_two = (3.0 * latitude**2 - 1.0) / 2.0 * np.ones((1, grid.nlon))
    field = xr.DataArray(
        degree_two,
        dims=("lat", "lon"),
        coords={"lat": grid.latitude, "lon": grid.longitude},
    )

    tapered = sg.filter(field, "T2", taper=0.1)

    np.testing.assert_allclose(tapered, 0.1 * field, rtol=0.0, atol=5.0e-16)


def test_spectral_error_cases_are_explicit() -> None:
    field = scalar_field(supported_grid("cc"))

    with pytest.raises(ValueError, match="either truncation"):
        sg.filter(field, "T2", lmin=0, lmax=2)
    with pytest.raises(ValueError, match="both lmin"):
        sg.filter(field, lmax=2)
    with pytest.raises(ValueError, match="taper"):
        sg.filter(field, "T2", taper=0.0)
    with pytest.raises(ValueError, match="exceeds"):
        sg.filter(field, "T99")
    with pytest.raises(ValueError, match="exceeds"):
        sg.regrid(field, sg.clenshaw_curtis_grid(9, 18), truncation="T8")
    with pytest.raises(ValueError, match="must be distinct"):
        sg.gradient(field, eastward="gradient", northward="gradient")
