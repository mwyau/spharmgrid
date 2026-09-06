"""Analytic fields shared by spharmgrid's numerical tests."""

from __future__ import annotations

import inspect
from typing import Literal

import numpy as np
import pytest
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


@pytest.fixture(autouse=True)
def report_assert_allclose_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report numerical error floors without failing calibration comparisons."""

    def report_allclose(
        actual: object,
        desired: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        del args, kwargs
        actual_values, desired_values = np.broadcast_arrays(
            np.asarray(actual), np.asarray(desired)
        )
        difference = np.abs(actual_values - desired_values)
        finite = (
            np.isfinite(difference)
            & np.isfinite(actual_values)
            & np.isfinite(desired_values)
        )

        if np.any(finite):
            max_abs = float(np.max(difference[finite]))
            scale = float(np.max(np.abs(desired_values[finite])))
            near_zero_threshold = max(scale * 1.0e-12, np.finfo(np.float64).tiny)
            near_zero = finite & (np.abs(desired_values) <= near_zero_threshold)
            relative = finite & (np.abs(desired_values) > near_zero_threshold)
            max_near_zero_abs = (
                float(np.max(difference[near_zero])) if np.any(near_zero) else 0.0
            )
            max_rel = (
                float(
                    np.max(
                        difference[relative] / np.abs(desired_values[relative])
                    )
                )
                if np.any(relative)
                else 0.0
            )
        else:
            max_abs = 0.0
            max_near_zero_abs = 0.0
            max_rel = 0.0

        source = "unknown:0"
        for frame in inspect.stack()[1:]:
            normalized = frame.filename.replace("\\", "/")
            if "/tests/" in normalized and not normalized.endswith("tests/conftest.py"):
                source = f"tests/{normalized.split('/tests/', 1)[1]}:{frame.lineno}"
                break

        print(
            "CALIBRATE "
            f"{source} "
            f"max_abs={max_abs:.17e} "
            f"near_zero_abs={max_near_zero_abs:.17e} "
            f"max_rel={max_rel:.17e}"
        )

    monkeypatch.setattr(np.testing, "assert_allclose", report_allclose)
