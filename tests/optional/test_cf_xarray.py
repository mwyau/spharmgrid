"""Optional cf-xarray coordinate-discovery integration checks."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import spharmgrid as sg


def test_cf_xarray_axis_metadata_can_identify_supported_coordinates() -> None:
    """Axis/units metadata is an optional fallback after the core paths."""
    pytest.importorskip("cf_xarray")
    latitude = xr.DataArray(
        np.linspace(-90.0, 90.0, 17),
        dims="y",
        attrs={"axis": "Y", "units": "degrees_north"},
    )
    longitude = xr.DataArray(
        np.linspace(0.0, 360.0, 36, endpoint=False),
        dims="x",
        attrs={"axis": "X", "units": "degrees_east"},
    )
    field = xr.DataArray(
        np.zeros((17, 36)),
        dims=("y", "x"),
        coords={"y": latitude, "x": longitude},
    )

    assert sg.detect_grid(field).kind == "cc"
