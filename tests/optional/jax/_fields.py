# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Analytic fields used by the S2FFT/JAX tests."""

from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr

import spharmgrid as sg

GridKind = Literal["gl", "gl_2l", "cc"]


def make_grid(
    kind: GridKind,
    *,
    nlat: int,
    latitude_order: Literal["ascending", "descending"] = "ascending",
    lon0: float = 0.0,
) -> sg.Grid:
    """Construct a GL or CC grid for backend parity tests."""
    if kind == "gl":
        return sg.gaussian_grid(
            nlat,
            2 * nlat - 1,
            latitude_order=latitude_order,
            lon0=lon0,
        )
    if kind == "gl_2l":
        return sg.gaussian_grid(
            nlat,
            2 * nlat,
            latitude_order=latitude_order,
            lon0=lon0,
        )
    return sg.clenshaw_curtis_grid(
        nlat,
        2 * (nlat - 1),
        latitude_order=latitude_order,
        lon0=lon0,
    )


def scalar_values(grid: sg.Grid) -> np.ndarray:
    """Return resolved scalar content through total degree four."""
    latitude = np.deg2rad(grid.latitude)[:, None]
    longitude = np.deg2rad(grid.longitude)[None, :]
    sine = np.sin(latitude)
    cosine = np.cos(latitude)
    values = (
        0.8
        + 0.35 * sine
        + 0.25 * (3.0 * sine**2 - 1.0) / 2.0
        + 0.20 * cosine * np.cos(longitude)
        + 0.17 * sine * cosine * np.sin(longitude)
        + 0.13 * cosine**2 * np.cos(2.0 * longitude)
        + 0.11 * sine * cosine**2 * np.sin(2.0 * longitude)
        + 0.07 * cosine**3 * np.cos(3.0 * longitude)
        + 0.05 * (5.0 * sine**3 - 3.0 * sine) / 2.0
        + 0.03 * cosine**4 * np.sin(4.0 * longitude)
    )
    return np.broadcast_to(values, (grid.nlat, grid.nlon)).copy()


def scalar_potentials(grid: sg.Grid) -> tuple[np.ndarray, np.ndarray]:
    """Return smooth velocity-potential and streamfunction test fields."""
    latitude = np.deg2rad(grid.latitude)[:, None]
    longitude = np.deg2rad(grid.longitude)[None, :]
    cosine = np.cos(latitude)
    velocity_potential = 1.4 * cosine * np.cos(longitude) + 0.6 * cosine**2 * np.cos(
        2.0 * longitude
    )
    streamfunction = 1.2 * cosine * np.sin(longitude) + 0.8 * cosine**2 * np.sin(
        2.0 * longitude
    )
    shape = (grid.nlat, grid.nlon)
    return (
        np.broadcast_to(velocity_potential, shape).copy(),
        np.broadcast_to(streamfunction, shape).copy(),
    )


def vector_values(grid: sg.Grid, radius: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return a mixed gradient and rotated-gradient field.

    The scalar potentials contain valid degree-one and degree-two spherical
    harmonic factors.  Their geographic components are written analytically
    so the vector tests do not rely on a transform round trip to establish the
    spin-1 signs.
    """
    latitude = np.deg2rad(grid.latitude)[:, None]
    longitude = np.deg2rad(grid.longitude)[None, :]
    sine = np.sin(latitude)
    cosine = np.cos(latitude)
    sine_longitude = np.sin(longitude)
    cosine_longitude = np.cos(longitude)
    sine_two_longitude = np.sin(2.0 * longitude)
    cosine_two_longitude = np.cos(2.0 * longitude)

    divergent_eastward = (
        -1.4 * sine_longitude - 1.2 * cosine * sine_two_longitude
    ) / radius
    divergent_northward = (
        -1.4 * sine * cosine_longitude - 1.2 * sine * cosine * cosine_two_longitude
    ) / radius
    rotational_eastward = (
        1.2 * sine * sine_longitude + 1.6 * sine * cosine * sine_two_longitude
    ) / radius
    rotational_northward = (
        1.2 * cosine_longitude + 1.6 * cosine * cosine_two_longitude
    ) / radius

    shape = (grid.nlat, grid.nlon)
    return (
        np.broadcast_to(divergent_eastward + rotational_eastward, shape).copy(),
        np.broadcast_to(divergent_northward + rotational_northward, shape).copy(),
    )


def pure_gradient_values(
    grid: sg.Grid, radius: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return the gradient of the analytic velocity-potential field."""
    latitude = np.deg2rad(grid.latitude)[:, None]
    longitude = np.deg2rad(grid.longitude)[None, :]
    sine = np.sin(latitude)
    cosine = np.cos(latitude)
    eastward = (
        -1.4 * np.sin(longitude) - 1.2 * cosine * np.sin(2.0 * longitude)
    ) / radius
    northward = (
        -1.4 * sine * np.cos(longitude) - 1.2 * sine * cosine * np.cos(2.0 * longitude)
    ) / radius
    shape = (grid.nlat, grid.nlon)
    return np.broadcast_to(eastward, shape).copy(), np.broadcast_to(
        northward, shape
    ).copy()


def pure_rotational_values(
    grid: sg.Grid, radius: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return the rotated gradient of the analytic streamfunction."""
    latitude = np.deg2rad(grid.latitude)[:, None]
    longitude = np.deg2rad(grid.longitude)[None, :]
    sine = np.sin(latitude)
    cosine = np.cos(latitude)
    eastward = (
        1.2 * sine * np.sin(longitude) + 1.6 * sine * cosine * np.sin(2.0 * longitude)
    ) / radius
    northward = (
        1.2 * np.cos(longitude) + 1.6 * cosine * np.cos(2.0 * longitude)
    ) / radius
    shape = (grid.nlat, grid.nlon)
    return np.broadcast_to(eastward, shape).copy(), np.broadcast_to(
        northward, shape
    ).copy()


def as_xarray(values: np.ndarray, grid: sg.Grid, name: str = "field") -> xr.DataArray:
    """Wrap values with the grid coordinates for DUCC comparisons."""
    return xr.DataArray(
        values,
        dims=("lat", "lon"),
        coords={"lat": grid.latitude, "lon": grid.longitude},
        name=name,
    )
