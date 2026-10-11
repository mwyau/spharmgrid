# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Grid recognition and coordinate-representation tests."""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
import pytest
import xarray as xr
from numpy.typing import NDArray

import spharmgrid as sg
from tests.conftest import SupportedGridKind, scalar_field, supported_grid


@pytest.mark.parametrize("kind", ["cc", "gl", "f1"])
@pytest.mark.parametrize("latitude_order", ["ascending", "descending"])
def test_constructed_grid_is_detected(
    kind: SupportedGridKind,
    latitude_order: Literal["ascending", "descending"],
) -> None:
    grid = supported_grid(kind, latitude_order=latitude_order)
    field = scalar_field(grid)

    detected = sg.detect_grid(field)

    assert detected.kind == kind
    np.testing.assert_array_equal(detected.latitude, grid.latitude)
    np.testing.assert_array_equal(detected.longitude, grid.longitude)


@pytest.mark.parametrize("kind", ["cc", "gl", "f1"])
@pytest.mark.parametrize("longitude_origin", [0.0, -180.0])
def test_cyclic_longitude_conventions_preserve_the_physical_field(
    kind: SupportedGridKind, longitude_origin: float
) -> None:
    grid = supported_grid(kind, lon0=longitude_origin)
    field = scalar_field(grid)

    result = sg.filter(field, "T3")

    np.testing.assert_allclose(result, field, rtol=0.0, atol=1.0e-14)
    np.testing.assert_array_equal(result.lon, field.lon)


@pytest.mark.parametrize("kind", ["cc", "gl", "f1"])
def test_cyclic_longitude_roll_preserves_phase_and_restores_coordinates(
    kind: SupportedGridKind,
) -> None:
    grid = supported_grid(kind)
    rolled_grid = sg.Grid(
        grid.kind,
        grid.latitude,
        np.roll(grid.longitude, 5),
    )
    rolled_field = scalar_field(rolled_grid)

    result = sg.regrid(rolled_field, grid, "T3")

    expected = scalar_field(grid)
    np.testing.assert_array_equal(result.lat, expected.lat)
    np.testing.assert_array_equal(result.lon, expected.lon)
    np.testing.assert_allclose(result, expected, rtol=0.0, atol=2.0e-13)


def test_cf_coordinate_metadata_precedes_canonical_name() -> None:
    grid = supported_grid("cc")
    field = scalar_field(grid).rename({"lat": "y", "lon": "x"})
    field = field.assign_coords(
        y=xr.DataArray(
            grid.latitude,
            dims="y",
            attrs={"standard_name": "latitude", "units": "degrees_north"},
        ),
        x=xr.DataArray(
            grid.longitude,
            dims="x",
            attrs={"standard_name": "longitude", "units": "degrees_east"},
        ),
    )

    detected = sg.detect_grid(field)

    assert detected.kind == "cc"


def test_ambiguous_coordinate_names_raise() -> None:
    grid = supported_grid("cc")
    field = scalar_field(grid).assign_coords(latitude=("lat", grid.latitude))

    with pytest.raises(ValueError, match="ambiguous latitude coordinate"):
        sg.detect_grid(field)


@pytest.mark.parametrize(
    ("latitude", "longitude", "message"),
    [
        (
            np.linspace(-80.0, 80.0, 17),
            np.linspace(0.0, 360.0, 36, endpoint=False),
            "unsupported latitude",
        ),
        (
            np.linspace(-90.0, 90.0, 17),
            np.linspace(0.0, 360.0, 37),
            "duplicated cyclic endpoint",
        ),
        (
            np.linspace(-90.0, 90.0, 17),
            np.array([0.0, 10.0, 30.0, 40.0]),
            "uniformly spaced",
        ),
    ],
)
def test_unsupported_grid_coordinates_raise(
    latitude: NDArray[np.float64],
    longitude: NDArray[np.float64],
    message: str,
) -> None:
    field = xr.DataArray(
        np.zeros((latitude.size, longitude.size)),
        dims=("lat", "lon"),
        coords={"lat": latitude, "lon": longitude},
    )

    with pytest.raises(ValueError, match=message):
        sg.detect_grid(field)


def test_generated_target_coordinates_carry_cf_axis_metadata() -> None:
    source = scalar_field(supported_grid("cc"))
    target = sg.fejer1_grid(12, 24)

    result = sg.regrid(source, target)

    assert result.lat.attrs == {
        "standard_name": "latitude",
        "units": "degrees_north",
        "axis": "Y",
    }
    assert result.lon.attrs == {
        "standard_name": "longitude",
        "units": "degrees_east",
        "axis": "X",
    }


@pytest.mark.parametrize("latitude_order", ["descending", "ascending"])
def test_fejer1_nodes_match_definition(
    latitude_order: Literal["ascending", "descending"],
) -> None:
    grid = sg.fejer1_grid(
        8,
        16,
        lon0=37.5,
        latitude_order=latitude_order,
    )
    north_to_south = 90.0 - np.rad2deg(
        np.pi * (np.arange(8, dtype=np.float64) + 0.5) / 8
    )
    expected = (
        north_to_south if latitude_order == "descending" else north_to_south[::-1]
    )

    assert grid.kind == "f1"
    np.testing.assert_allclose(grid.latitude, expected, rtol=0.0, atol=1.0e-14)
    np.testing.assert_array_equal(grid.longitude, 37.5 + np.arange(16) * 22.5)


@pytest.mark.parametrize(
    ("nlat", "nlon", "latitude_order", "message"),
    [
        (1, 16, "ascending", "nlat"),
        (8, 1, "ascending", "nlon"),
        (8, 16, "sideways", "latitude_order"),
    ],
)
def test_fejer1_constructor_rejects_invalid_sampling(
    nlat: int,
    nlon: int,
    latitude_order: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        sg.fejer1_grid(
            nlat,
            nlon,
            latitude_order=cast(Literal["ascending", "descending"], latitude_order),
        )


def test_fejer1_detection_rejects_near_miss_and_partial_latitudes() -> None:
    grid = sg.fejer1_grid(8, 16)
    near_miss = grid.latitude.copy()
    near_miss[2] += 0.01
    for latitude in (near_miss, grid.latitude[1:]):
        field = xr.DataArray(
            np.zeros((latitude.size, grid.nlon)),
            dims=("lat", "lon"),
            coords={"lat": latitude, "lon": grid.longitude},
        )
        with pytest.raises(ValueError, match="unsupported latitude"):
            sg.detect_grid(field)
