"""Analytic fields shared by spharmgrid's numerical tests."""

from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr

import spharmgrid as sg


def supported_grid(
    kind: Literal["cc", "gl"],
    *,
    latitude_order: Literal["ascending", "descending"] = "ascending",
    lon0: float = 0.0,
) -> sg.Grid:
    """Return a modest grid that resolves deterministic low-degree fields."""
    if kind == "cc":
        return sg.clenshaw_curtis_grid(
            17,
            36,
            latitude_order=latitude_order,
            lon0=lon0,
        )
    return sg.gaussian_grid(
        16,
        36,
        latitude_order=latitude_order,
        lon0=lon0,
    )


def scalar_field(
    grid: sg.Grid,
    *,
    name: str = "field",
    leading: bool = False,
) -> xr.DataArray:
    """A deterministic combination of degree 0, 1, 2, and m=2 content."""
    latitude = np.deg2rad(grid.latitude)[:, None]
    longitude = np.deg2rad(grid.longitude)[None, :]
    sin_latitude = np.sin(latitude)
    values = (
        1.25
        + 0.5 * sin_latitude
        + 0.75 * (3.0 * sin_latitude**2 - 1.0) / 2.0
        + 0.2 * np.cos(latitude) ** 2 * np.cos(2.0 * longitude)
    )
    data = xr.DataArray(
        values,
        dims=("lat", "lon"),
        coords={"lat": grid.latitude, "lon": grid.longitude},
        name=name,
        attrs={"units": "K", "long_name": "Analytic scalar field"},
    )
    if not leading:
        return data
    return xr.concat(
        [data, 2.0 * data],
        dim=xr.DataArray(
            np.array(["first", "second"], dtype=object), dims="member", name="member"
        ),
    )


def degree_one_field(grid: sg.Grid, *, name: str = "field") -> xr.DataArray:
    """The degree-one axisymmetric harmonic sin(latitude)."""
    values = np.sin(np.deg2rad(grid.latitude))[:, None] * np.ones(
        (1, grid.nlon), dtype=np.float64
    )
    return xr.DataArray(
        values,
        dims=("lat", "lon"),
        coords={"lat": grid.latitude, "lon": grid.longitude},
        name=name,
    )


def solid_body_wind(
    grid: sg.Grid, amplitude: float = 10.0
) -> tuple[xr.DataArray, xr.DataArray]:
    """Eastward solid-body rotation with known vorticity and zero divergence."""
    shape = (grid.nlat, grid.nlon)
    u = (
        amplitude
        * np.cos(np.deg2rad(grid.latitude))[:, None]
        * np.ones((1, grid.nlon), dtype=np.float64)
    )
    v = np.zeros(shape, dtype=np.float64)
    coordinates = {"lat": grid.latitude, "lon": grid.longitude}
    return (
        xr.DataArray(u, dims=("lat", "lon"), coords=coordinates, name="u"),
        xr.DataArray(v, dims=("lat", "lon"), coords=coordinates, name="v"),
    )
