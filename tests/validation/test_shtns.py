# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optional F1 scalar and vector parity checks against SHTns on its regular DCT grid."""

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


def test_shtns_f1_grid_and_scalar_conventions_at_t35() -> None:
    special = pytest.importorskip("scipy.special")
    shtns = pytest.importorskip("shtns")

    nlat, nlon, degree, order = 72, 144, 35, 35
    grid = sg.fejer1_grid(nlat, nlon, latitude_order="descending")
    transform = shtns.sht(degree, degree, 1, shtns.sht_orthonormal)
    transform.set_grid(
        nlat,
        nlon,
        shtns.sht_reg_dct | shtns.SHT_PHI_CONTIGUOUS,
        1.0e-10,
    )

    shtns_latitude = np.rad2deg(np.arcsin(transform.cos_theta))
    np.testing.assert_allclose(
        shtns_latitude,
        grid.latitude,
        rtol=0.0,
        atol=2.0e-13,
    )

    theta = np.deg2rad(90.0 - grid.latitude)[:, None]
    phi = np.deg2rad(grid.longitude)[None, :]
    coefficient = 0.2 - 0.1j
    harmonic = special.sph_harm_y(degree, order, theta, phi)
    values = 2.0 * np.real(coefficient * harmonic)
    field = xr.DataArray(
        values,
        dims=("lat", "lon"),
        coords={"lat": grid.latitude, "lon": grid.longitude},
    )

    shtns_coefficients = transform.analys(values)
    shtns_index = transform.idx(degree, order)
    np.testing.assert_allclose(
        shtns_coefficients[shtns_index], coefficient, rtol=0.0, atol=2.0e-12
    )
    np.testing.assert_allclose(
        transform.synth(shtns_coefficients), values, rtol=0.0, atol=2.0e-11
    )

    ducc = sg.analyze(field, "T35")
    mode_index = np.flatnonzero(
        (alm_degrees(degree, degree) == degree) & (alm_orders(degree, degree) == order)
    )
    assert mode_index.size == 1
    ducc_coefficients = ducc._coefficients.values
    np.testing.assert_allclose(
        ducc_coefficients[mode_index[0]],
        shtns_coefficients[shtns_index],
        rtol=0.0,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(ducc.synthesize(), values, rtol=0.0, atol=2.0e-11)


def test_shtns_f1_vector_component_and_coefficient_conventions() -> None:
    special = pytest.importorskip("scipy.special")
    shtns = pytest.importorskip("shtns")

    nlat, nlon, degree, order = 72, 144, 3, 2
    grid = sg.fejer1_grid(nlat, nlon, latitude_order="descending")
    transform = shtns.sht(degree, degree, 1, shtns.sht_orthonormal)
    transform.set_grid(
        nlat,
        nlon,
        shtns.sht_reg_dct | shtns.SHT_PHI_CONTIGUOUS,
        1.0e-10,
    )
    assert (transform.nlat, transform.nphi) == (nlat, nlon)

    theta = np.deg2rad(90.0 - grid.latitude)[:, None]
    phi = np.deg2rad(grid.longitude)[None, :]
    coefficient = 0.2 - 0.1j
    _, derivatives = special.sph_harm_y(
        degree,
        order,
        theta,
        phi,
        diff_n=1,
    )
    gradient_east = 2.0 * np.real(coefficient * derivatives[..., 1]) / np.sin(theta)
    gradient_north = -2.0 * np.real(coefficient * derivatives[..., 0])
    vector_cases = (
        (gradient_east, gradient_north),
        (-gradient_north, gradient_east),
    )
    coordinates = {"lat": grid.latitude, "lon": grid.longitude}
    mode_index = np.flatnonzero(
        (alm_degrees(degree, degree) == degree) & (alm_orders(degree, degree) == order)
    )
    assert mode_index.size == 1
    normalization = np.sqrt(degree * (degree + 1))

    for eastward, northward in vector_cases:
        u = xr.DataArray(eastward, dims=("lat", "lon"), coords=coordinates)
        v = xr.DataArray(northward, dims=("lat", "lon"), coords=coordinates)
        vector = sg.analyze_vector(u, v, f"T{degree}")
        ducc_coefficients = (
            vector._coefficients.values[:, mode_index[0]] / normalization
        )

        # SHTns uses (v_theta, v_phi)=(-v_north, u_east). Its toroidal
        # coefficient has the opposite sign to DUCC's B coefficient.
        shtns_spheroidal, shtns_toroidal = transform.analys(-northward, eastward)
        shtns_index = transform.idx(degree, order)
        np.testing.assert_allclose(
            ducc_coefficients[0],
            shtns_spheroidal[shtns_index],
            rtol=0.0,
            atol=2.0e-13,
        )
        np.testing.assert_allclose(
            ducc_coefficients[1],
            -shtns_toroidal[shtns_index],
            rtol=0.0,
            atol=2.0e-13,
        )

        shtns_theta, shtns_phi = transform.synth(shtns_spheroidal, shtns_toroidal)
        np.testing.assert_allclose(shtns_phi, eastward, rtol=0.0, atol=2.0e-12)
        np.testing.assert_allclose(-shtns_theta, northward, rtol=0.0, atol=2.0e-12)
        ducc_eastward, ducc_northward = vector.synthesize()
        np.testing.assert_allclose(ducc_eastward, eastward, rtol=0.0, atol=2.0e-12)
        np.testing.assert_allclose(ducc_northward, northward, rtol=0.0, atol=2.0e-12)
