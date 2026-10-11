# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optional S2FFT DH comparison on the shared sub-bandlimit of an F1 grid."""

from __future__ import annotations

import os

import numpy as np
import pytest
import xarray as xr

import spharmgrid as sg
from spharmgrid._ducc import alm_degrees, alm_orders
from tests.validation.test_scipy import _harmonic_scalar

pytestmark = pytest.mark.skipif(
    os.environ.get("SPHARMGRID_RUN_VALIDATION") != "1",
    reason="run only in the optional validation workflow",
)


def test_s2fft_dh_coefficients_match_ducc_on_the_f1_subbandlimit() -> None:
    s2fft = pytest.importorskip("s2fft")
    from s2fft.sampling import s2_samples

    bandlimit = 16
    grid = sg.fejer1_grid(
        2 * bandlimit,
        2 * bandlimit - 1,
        latitude_order="descending",
    )
    theta = np.deg2rad(90.0 - grid.latitude)
    phi = np.mod(np.deg2rad(grid.longitude), 2.0 * np.pi)
    np.testing.assert_allclose(
        theta,
        s2_samples.thetas(bandlimit, sampling="dh"),
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        phi,
        s2_samples.phis_equiang(bandlimit, sampling="dh"),
        rtol=0.0,
        atol=1.0e-14,
    )

    components = (
        _harmonic_scalar(grid, 15, 0, 0.3),
        _harmonic_scalar(grid, 15, 7, -0.13 + 0.07j),
        _harmonic_scalar(grid, 15, 15, 0.2 - 0.1j),
    )
    values = sum(components)
    field = xr.DataArray(
        values,
        dims=("lat", "lon"),
        coords={"lat": grid.latitude, "lon": grid.longitude},
    )

    ducc = sg.analyze(field, "T15")
    s2fft_coefficients = s2fft.forward(
        values,
        L=bandlimit,
        spin=0,
        sampling="dh",
        method="numpy",
        reality=False,
    )
    degrees = alm_degrees(15, 15)
    orders = alm_orders(15, 15)
    expected = s2fft_coefficients[degrees, orders + bandlimit - 1]

    assert s2fft_coefficients.shape == (bandlimit, 2 * bandlimit - 1)
    np.testing.assert_allclose(
        ducc._coefficients.values,
        expected,
        rtol=0.0,
        atol=3.0e-12,
    )
