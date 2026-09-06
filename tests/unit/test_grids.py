"""Grid recognition and coordinate-representation tests."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
import xarray as xr

import spharmgrid as sg
from tests.conftest import scalar_field, supported_grid


@pytest.mark.parametrize("kind", ["cc", "gl"])
@pytest.mark.parametrize("latitude_order", ["ascending", "descending"])
def test_constructed_grid_is_detected(
    kind: Literal["cc", "gl"],
    latitude_order: Literal["ascending", "descending"],
) -> None:
    grid = supported_grid(kind, latitude_order=latitude_order)
    field = scalar_field(grid)

    detected = sg.detect_grid(field)

    assert detected.kind == kind
    np.testing.assert_array_equal(detected.latitude, grid.latitude)
    np.testing.assert_array_equal(detected.longitude, grid.longitude)
    assert field.sg.grid_type == kind


@pytest.mark.parametrize("kind", ["cc", "gl"])
@pytest.mark.parametrize("longitude_origin", [0.0, -180.0])
def test_cyclic_longitude_conventions_preserve_the_physical_field(
    kind: Literal["cc", "gl"], longitude_origin: float
) -> None:
    grid = supported_grid(kind, lon0=longitude_origin)
    field = scalar_field(grid)

    result = sg.filter(field, "T3")

    np.testing.assert_allclose(result, field, rtol=0.0, atol=1.0e-14)
    np.testing.assert_array_equal(result.lon, field.lon)


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
    latitude: np.ndarray, longitude: np.ndarray, message: str
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
    target = sg.gaussian_grid(12, 24)

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
