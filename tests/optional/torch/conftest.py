# SPDX-FileCopyrightText: Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixtures for the optional torch-harmonics backend tests."""

from __future__ import annotations

import importlib

import numpy as np
import pytest
import xarray as xr

import spharmgrid as sg

torch = pytest.importorskip("torch")
pytest.importorskip("torch_harmonics")

importlib.import_module("spharmgrid.torch")
importlib.import_module("spharmgrid.torch.nn")


def make_fields(
    grid: sg.Grid,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a scalar field with known modes through degree two and a wind."""
    latitude = np.deg2rad(grid.latitude)[:, None]
    longitude = np.deg2rad(grid.longitude)[None, :]
    sine = np.sin(latitude)
    cosine = np.cos(latitude)
    scalar = (
        1.25
        + 0.5 * sine
        + 0.75 * (3.0 * sine**2 - 1.0) / 2.0
        + 0.2 * cosine**2 * np.cos(2.0 * longitude)
    )
    eastward = 4.0 * cosine + 0.3 * np.sin(longitude)
    northward = 7.0 * cosine - 0.2 * sine * np.cos(longitude)
    return tuple(
        torch.as_tensor(values, dtype=dtype) for values in (scalar, eastward, northward)
    )


def make_nonaxisymmetric_wind(
    grid: sg.Grid,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the unit-radius ``l=1, m=1`` wind from two analytic potentials."""
    latitude = np.deg2rad(grid.latitude)[:, None]
    longitude = np.deg2rad(grid.longitude)[None, :]
    eastward = -np.sin(longitude) + np.sin(latitude) * np.sin(longitude)
    northward = -np.sin(latitude) * np.cos(longitude) + np.cos(longitude)
    return tuple(
        torch.as_tensor(values, dtype=dtype) for values in (eastward, northward)
    )


def as_xarray(field: torch.Tensor, grid: sg.Grid, name: str = "field") -> xr.DataArray:
    """Put a test tensor at the explicit comparison boundary."""
    return xr.DataArray(
        field.detach().cpu().numpy(),
        dims=("lat", "lon"),
        coords={"lat": grid.latitude, "lon": grid.longitude},
        name=name,
    )


@pytest.fixture
def gl_grid() -> sg.Grid:
    return sg.gaussian_grid(8, 18, latitude_order="ascending", lon0=45.0)


@pytest.fixture
def cc_grid() -> sg.Grid:
    return sg.clenshaw_curtis_grid(
        17,
        36,
        latitude_order="ascending",
        lon0=45.0,
    )
