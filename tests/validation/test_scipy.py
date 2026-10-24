# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""SciPy spherical harmonic checks for DUCC0 Fejér-1 transforms."""

from __future__ import annotations

import os

import numpy as np
import pytest
import xarray as xr

import spharmgrid as sg
from spharmgrid._ducc import alm_degrees, alm_orders

pytestmark = pytest.mark.skipif(
    os.environ.get("SPHARMGRID_RUN_VALIDATION") != "1",
    reason="run only in the optional validation workflow",
)


def _f1_grid(nlat: int, nlon: int) -> sg.Grid:
    return sg.fejer1_grid(nlat, nlon, latitude_order="descending")


def _harmonic_component(
    grid: sg.Grid,
    degree: int,
    order: int,
    coefficient: complex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a real harmonic and its eastward/northward unit-sphere gradient."""
    special = pytest.importorskip("scipy.special")
    theta = np.deg2rad(90.0 - grid.latitude)[:, None]
    phi = np.mod(np.deg2rad(grid.longitude), 2.0 * np.pi)[None, :]
    harmonic, derivatives = special.sph_harm_y(
        degree,
        order,
        theta,
        phi,
        diff_n=1,
    )
    scaled = coefficient * harmonic
    dtheta = coefficient * derivatives[..., 0]
    dphi = coefficient * derivatives[..., 1]
    if order:
        value = 2.0 * np.real(scaled)
        dtheta = 2.0 * np.real(dtheta)
        dphi = 2.0 * np.real(dphi)
    else:
        value = np.real(scaled)
        dtheta = np.real(dtheta)
        dphi = np.real(dphi)
    return value, dphi / np.sin(theta), -dtheta


def _high_degree_fields(
    grid: sg.Grid,
    degree: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    zonal = _harmonic_component(grid, degree, 0, 0.3)
    tesseral = _harmonic_component(grid, degree, degree // 2, -0.13 + 0.07j)
    sectoral = _harmonic_component(grid, degree, degree, 0.2 - 0.1j)
    return (
        zonal[0] + tesseral[0] + sectoral[0],
        zonal[1] + tesseral[1] + sectoral[1],
        zonal[2] + tesseral[2] + sectoral[2],
    )


@pytest.mark.parametrize(
    ("nlat", "nlon", "degree"),
    [(72, 144, 71), (180, 360, 179)],
)
def test_f1_scalar_and_spin1_high_degree_against_scipy(
    nlat: int,
    nlon: int,
    degree: int,
) -> None:
    grid = _f1_grid(nlat, nlon)
    scalar_values, expected_east, expected_north = _high_degree_fields(grid, degree)
    field = xr.DataArray(
        scalar_values,
        dims=("lat", "lon"),
        coords={"lat": grid.latitude, "lon": grid.longitude},
        name="analytic_scalar",
    )

    spectral = sg.analyze(field, f"T{degree}")
    assert (spectral.spec.lmax, spectral.spec.mmax) == (degree, degree)
    degrees = alm_degrees(degree, degree)
    orders = alm_orders(degree, degree)
    coefficients = spectral._coefficients.values
    for order, expected in (
        (0, 0.3),
        (degree // 2, -0.13 + 0.07j),
        (degree, 0.2 - 0.1j),
    ):
        mode_index = np.flatnonzero((degrees == degree) & (orders == order))
        assert mode_index.size == 1
        np.testing.assert_allclose(
            coefficients[mode_index[0]], expected, rtol=0.0, atol=3.0e-14
        )
    scalar_tolerance = 4.0e-11 if degree == 179 else 2.0e-12
    np.testing.assert_allclose(
        spectral.synthesize(),
        scalar_values,
        rtol=0.0,
        atol=scalar_tolerance,
    )
    np.testing.assert_allclose(
        sg.filter(field, f"T{degree - 1}"),
        0.0,
        rtol=0.0,
        atol=scalar_tolerance,
    )

    gradient = sg.gradient(field, radius=1.0)
    np.testing.assert_allclose(
        gradient.gradient_eastward,
        expected_east,
        rtol=0.0,
        atol=4.0e-10 if degree == 179 else 2.0e-11,
    )
    np.testing.assert_allclose(
        gradient.gradient_northward,
        expected_north,
        rtol=0.0,
        atol=4.0e-10 if degree == 179 else 2.0e-11,
    )

    vector = sg.analyze_vector(
        xr.DataArray(
            expected_east,
            dims=("lat", "lon"),
            coords=field.coords,
            name="u",
        ),
        xr.DataArray(
            expected_north,
            dims=("lat", "lon"),
            coords=field.coords,
            name="v",
        ),
        f"T{degree}",
    )
    assert (vector.spec.lmax, vector.spec.mmax) == (degree, degree)
    reconstructed_east, reconstructed_north = vector.synthesize()
    np.testing.assert_allclose(
        reconstructed_east,
        expected_east,
        rtol=0.0,
        atol=4.0e-10 if degree == 179 else 2.0e-11,
    )
    np.testing.assert_allclose(
        reconstructed_north,
        expected_north,
        rtol=0.0,
        atol=4.0e-10 if degree == 179 else 2.0e-11,
    )


def test_f1_longitude_limit_is_independent_of_latitude_limit() -> None:
    degree, order = 71, 69
    grid = _f1_grid(72, 140)
    scalar_values, _, _ = _harmonic_component(grid, degree, order, 0.4 + 0.2j)
    field = xr.DataArray(
        scalar_values,
        dims=("lat", "lon"),
        coords={"lat": grid.latitude, "lon": grid.longitude},
    )

    spectral = sg.analyze(field, f"T{degree}x{order}")
    assert (spectral.spec.lmax, spectral.spec.mmax) == (degree, order)
    np.testing.assert_allclose(
        spectral.synthesize(), scalar_values, rtol=0.0, atol=2.0e-12
    )
    with pytest.raises(ValueError, match=f"source longitude bandwidth {order}"):
        sg.analyze(field, f"T{degree}x{order + 1}")
    with pytest.raises(ValueError, match=f"source latitude bandwidth {degree}"):
        sg.analyze(field, f"T{degree + 1}x{order}")
