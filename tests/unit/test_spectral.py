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
    alm_subselection,
    geometry_for,
    scalar_synthesis,
)
from spharmgrid._transform import TransformSpec
from spharmgrid.grids import grid_layout
from spharmgrid.spectral import (
    _spectral_selection_is_within,
    apply_spectral_selection,
    resolve_transform_spec,
)
from tests.conftest import scalar_field, supported_grid


def _retained_modes(spec: TransformSpec) -> set[tuple[int, int]]:
    """Enumerate a small domain for the containment oracle only."""
    modes: set[tuple[int, int]] = set()
    for degree in range(spec.lmin, spec.lmax + 1):
        for order in range(min(degree, spec.mmax) + 1):
            if spec.truncation == "rhomboidal" and (
                degree - order > spec.lmax - spec.mmax
            ):
                continue
            modes.add((degree, order))
    return modes


def _brute_spectral_selection_is_within(
    analyzed: TransformSpec, selection: TransformSpec | None
) -> bool:
    """Reference containment predicate used only by the tests."""
    if selection is None:
        return True
    analyzed_modes = _retained_modes(analyzed)
    return _retained_modes(selection).issubset(analyzed_modes)


def _small_transform_specs() -> tuple[TransformSpec, ...]:
    specs: list[TransformSpec] = []
    for lmax in range(5):
        for lmin in range(lmax + 1):
            for mmax in range(lmax + 1):
                for truncation in ("triangular", "trapezoidal", "rhomboidal"):
                    if truncation == "triangular" and mmax != lmax:
                        continue
                    specs.append(TransformSpec(lmin, lmax, mmax, truncation))
    return tuple(specs)


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


def test_analytical_selection_containment_matches_brute_force_oracle() -> None:
    specs = _small_transform_specs()

    for analyzed in specs:
        for selection in specs:
            expected = _brute_spectral_selection_is_within(analyzed, selection)
            actual = _spectral_selection_is_within(analyzed, selection)
            assert actual == expected, (analyzed, selection)


@pytest.mark.parametrize(
    ("analyzed", "selection", "expected"),
    [
        ("T6", "T2-6", True),
        ("T6", "R3", True),
        ("T6x3", "R3", True),
        ("R3", "R3", True),
        ("R3", "R2", True),
        ("R3", "T6", False),
        ("R3", "T2-6", False),
        ("R3", "T6x3", False),
    ],
)
def test_selection_containment_regression_domains(
    analyzed: str, selection: str, expected: bool
) -> None:
    assert (
        _spectral_selection_is_within(
            sg.parse_spectral(analyzed), sg.parse_spectral(selection)
        )
        is expected
    )


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


@pytest.mark.parametrize(
    ("source_lmax", "source_mmax", "target_lmax", "target_mmax"),
    [
        (0, 0, 0, 0),
        (3, 3, 3, 3),
        (8, 6, 5, 6),
        (8, 8, 8, 3),
        (12, 9, 6, 4),
        (80, 60, 70, 20),
    ],
)
def test_alm_subselection_matches_packed_degree_order_mapping(
    source_lmax: int,
    source_mmax: int,
    target_lmax: int,
    target_mmax: int,
) -> None:
    """Check every returned position against DUCC's packed ``(l, m)`` order."""
    source_degrees = alm_degrees(source_lmax, source_mmax)
    source_orders = alm_orders(source_lmax, source_mmax)
    source_positions = {
        (int(degree), int(order)): position
        for position, (degree, order) in enumerate(
            zip(source_degrees, source_orders, strict=True)
        )
    }
    target_degrees = alm_degrees(target_lmax, target_mmax)
    target_orders = alm_orders(target_lmax, target_mmax)
    expected = np.asarray(
        [
            source_positions[(int(degree), int(order))]
            for degree, order in zip(target_degrees, target_orders, strict=True)
        ],
        dtype=np.intp,
    )

    actual = alm_subselection(
        source_lmax,
        source_mmax,
        target_lmax,
        target_mmax,
    )

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.intp
    assert not actual.flags.writeable


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
