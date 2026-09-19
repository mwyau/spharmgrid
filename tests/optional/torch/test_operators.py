# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optional Torch differential and wind-operator tests."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest

import spharmgrid as sg
import spharmgrid.torch as sgt
from tests.optional.torch.conftest import make_fields, torch


def _assert_close(
    actual: torch.Tensor,
    expected: np.ndarray,
    dtype: torch.dtype,
    *,
    tolerances: tuple[float, float] | None = None,
    rtol: float | None = None,
    atol: float | None = None,
) -> None:
    if tolerances is None:
        tolerances = _value_tolerances(dtype)
    if rtol is None:
        rtol = tolerances[0]
    if atol is None:
        atol = tolerances[1]
    np.testing.assert_allclose(
        actual.detach().cpu().numpy(),
        expected,
        rtol=rtol,
        atol=atol,
    )


def _value_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    """Return tolerances for dimensionless field and analytic comparisons."""
    return (1.0e-12, 1.0e-13) if dtype == torch.float64 else (5.0e-6, 2.0e-6)


def test_cc_full_state_operations_raise_capability_error(cc_grid: sg.Grid) -> None:
    field, eastward, northward = make_fields(cc_grid)
    error = (
        r"torch-harmonics.*CC triangular bands through T8.*"
        r"Full-domain operations.*filter, regrid, and regrid_vector"
    )

    with pytest.raises(ValueError, match=error):
        sgt.gradient(field, grid=cc_grid)
    with pytest.raises(ValueError, match=error):
        sgt.kinematics(eastward, northward, grid=cc_grid)


@pytest.mark.parametrize("latitude_order", ["ascending", "descending"])
def test_analytic_scalar_and_vector_identities(
    latitude_order: Literal["ascending", "descending"],
) -> None:
    grid = sg.gaussian_grid(16, 36, latitude_order=latitude_order, lon0=105.0)
    latitude = np.deg2rad(grid.latitude)[:, None]
    cosine = np.cos(latitude) * np.ones((1, grid.nlon))
    sine = np.sin(latitude) * np.ones((1, grid.nlon))
    radius = sg.EARTH_RADIUS_M
    scalar = torch.as_tensor(sine, dtype=torch.float64)

    eastward_gradient, northward_gradient = sgt.gradient(
        scalar,
        grid=grid,
        radius=radius,
    )
    _assert_close(eastward_gradient, np.zeros_like(cosine), torch.float64, atol=5.0e-20)
    _assert_close(
        northward_gradient,
        cosine / radius,
        torch.float64,
        atol=5.0e-20,
    )
    _assert_close(
        sgt.laplacian(scalar, grid=grid, radius=radius),
        -2.0 * sine / radius**2,
        torch.float64,
        atol=5.0e-26,
    )
    _assert_close(
        sgt.inverse_laplacian(scalar, grid=grid, radius=radius),
        -0.5 * radius**2 * sine,
        torch.float64,
        rtol=2.0e-14,
        atol=5.0e-9,
    )
    _assert_close(
        sgt.inverse_gradient(
            eastward_gradient,
            northward_gradient,
            grid=grid,
            radius=radius,
        ),
        sine,
        torch.float64,
        atol=5.0e-14,
    )

    rotational_amplitude = 10.0
    divergent_amplitude = 7.0
    eastward = torch.as_tensor(rotational_amplitude * cosine, dtype=torch.float64)
    northward = torch.as_tensor(divergent_amplitude * cosine, dtype=torch.float64)
    expected_vorticity = 2.0 * rotational_amplitude * sine / radius
    expected_divergence = -2.0 * divergent_amplitude * sine / radius
    vorticity, divergence = sgt.kinematics(eastward, northward, grid=grid)
    _assert_close(vorticity, expected_vorticity, torch.float64, atol=5.0e-20)
    _assert_close(divergence, expected_divergence, torch.float64, atol=5.0e-20)

    divergent_u, divergent_v, rotational_u, rotational_v = sgt.helmholtz(
        eastward,
        northward,
        grid=grid,
    )
    _assert_close(divergent_u, np.zeros_like(cosine), torch.float64, atol=5.0e-20)
    _assert_close(
        divergent_v, divergent_amplitude * cosine, torch.float64, atol=5.0e-14
    )
    _assert_close(
        rotational_u, rotational_amplitude * cosine, torch.float64, atol=5.0e-14
    )
    _assert_close(rotational_v, np.zeros_like(cosine), torch.float64, atol=5.0e-20)

    laplacian_u, laplacian_v = sgt.vector_laplacian(
        eastward,
        northward,
        grid=grid,
        radius=radius,
    )
    eigenvalue = -2.0 / radius**2
    _assert_close(
        laplacian_u, eigenvalue * eastward.numpy(), torch.float64, atol=5.0e-25
    )
    _assert_close(
        laplacian_v, eigenvalue * northward.numpy(), torch.float64, atol=5.0e-25
    )
    restored_u, restored_v = sgt.inverse_vector_laplacian(
        laplacian_u,
        laplacian_v,
        grid=grid,
        radius=radius,
    )
    _assert_close(restored_u, eastward.numpy(), torch.float64, atol=5.0e-14)
    _assert_close(restored_v, northward.numpy(), torch.float64, atol=5.0e-14)

    streamfunction, velocity_potential = sgt.potentials(
        eastward,
        northward,
        grid=grid,
        radius=radius,
    )
    _assert_close(
        streamfunction,
        -rotational_amplitude * radius * sine,
        torch.float64,
        rtol=1.0e-13,
        atol=1.0e-7,
    )
    _assert_close(
        velocity_potential,
        divergent_amplitude * radius * sine,
        torch.float64,
        rtol=1.0e-13,
        atol=1.0e-7,
    )

    recovered_rotational = sgt.rotational_wind(
        vorticity,
        grid=grid,
        source="vorticity",
        radius=radius,
    )
    recovered_divergent = sgt.divergent_wind(
        divergence,
        grid=grid,
        source="divergence",
        radius=radius,
    )
    _assert_close(
        recovered_rotational[0],
        eastward.numpy(),
        torch.float64,
        rtol=1.0e-13,
        atol=5.0e-14,
    )
    _assert_close(
        recovered_rotational[1],
        np.zeros_like(cosine),
        torch.float64,
        rtol=1.0e-13,
        atol=5.0e-20,
    )
    _assert_close(
        recovered_divergent[0],
        np.zeros_like(cosine),
        torch.float64,
        rtol=1.0e-13,
        atol=5.0e-20,
    )
    _assert_close(
        recovered_divergent[1],
        northward.numpy(),
        torch.float64,
        rtol=1.0e-13,
        atol=5.0e-14,
    )

    recovered_wind = sgt.wind(
        vorticity,
        divergence,
        grid=grid,
        source="vorticity_divergence",
        radius=radius,
    )
    _assert_close(
        recovered_wind[0],
        eastward.numpy(),
        torch.float64,
        rtol=1.0e-13,
        atol=5.0e-14,
    )
    _assert_close(
        recovered_wind[1],
        northward.numpy(),
        torch.float64,
        rtol=1.0e-13,
        atol=5.0e-14,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("latitude_order", ["ascending", "descending"])
def test_analytic_nonaxisymmetric_l1m1_vector_identities(
    latitude_order: Literal["ascending", "descending"],
    dtype: torch.dtype,
) -> None:
    """Check vector signs and phase against direct spherical-coordinate formulae."""
    grid = sg.gaussian_grid(8, 18, latitude_order=latitude_order, lon0=37.0)
    radius = 2.5
    latitude = np.deg2rad(grid.latitude)[:, None]
    longitude = np.deg2rad(grid.longitude)[None, :]
    sine_latitude = np.sin(latitude)
    cosine_latitude = np.cos(latitude)
    sine_longitude = np.sin(longitude)
    cosine_longitude = np.cos(longitude)
    shape = (grid.nlat, grid.nlon)
    sine_latitude = np.broadcast_to(sine_latitude, shape)
    cosine_latitude = np.broadcast_to(cosine_latitude, shape)
    sine_longitude = np.broadcast_to(sine_longitude, shape)
    cosine_longitude = np.broadcast_to(cosine_longitude, shape)

    chi_amplitude = 2.0
    psi_amplitude = -1.5
    chi = chi_amplitude * cosine_latitude * cosine_longitude
    psi = psi_amplitude * cosine_latitude * sine_longitude
    divergent_u = -chi_amplitude * sine_longitude / radius
    divergent_v = -chi_amplitude * sine_latitude * cosine_longitude / radius
    rotational_u = psi_amplitude * sine_latitude * sine_longitude / radius
    rotational_v = psi_amplitude * cosine_longitude / radius
    eastward = divergent_u + rotational_u
    northward = divergent_v + rotational_v
    tolerances = _value_tolerances(dtype)
    u = torch.as_tensor(eastward, dtype=dtype)
    v = torch.as_tensor(northward, dtype=dtype)
    scalar = torch.as_tensor(chi, dtype=dtype)

    actual_gradient = sgt.gradient(scalar, grid=grid, radius=radius)
    _assert_close(actual_gradient[0], divergent_u, dtype, tolerances=tolerances)
    _assert_close(actual_gradient[1], divergent_v, dtype, tolerances=tolerances)
    _assert_close(
        sgt.inverse_gradient(
            torch.as_tensor(divergent_u, dtype=dtype),
            torch.as_tensor(divergent_v, dtype=dtype),
            grid=grid,
            radius=radius,
        ),
        chi,
        dtype,
        tolerances=tolerances,
    )

    actual_vorticity, actual_divergence = sgt.kinematics(
        u,
        v,
        grid=grid,
        radius=radius,
    )
    _assert_close(
        actual_vorticity,
        -2.0 * psi / radius**2,
        dtype,
        tolerances=tolerances,
    )
    _assert_close(
        actual_divergence,
        -2.0 * chi / radius**2,
        dtype,
        tolerances=tolerances,
    )

    actual_potentials = sgt.potentials(u, v, grid=grid, radius=radius)
    _assert_close(actual_potentials[0], psi, dtype, tolerances=tolerances)
    _assert_close(actual_potentials[1], chi, dtype, tolerances=tolerances)

    actual_helmholtz = sgt.helmholtz(u, v, grid=grid, radius=radius)
    for actual, expected in zip(
        actual_helmholtz,
        (divergent_u, divergent_v, rotational_u, rotational_v),
        strict=True,
    ):
        _assert_close(actual, expected, dtype, tolerances=tolerances)

    for first, second, source in (
        (*actual_potentials, "potentials"),
        (actual_vorticity, actual_divergence, "vorticity_divergence"),
    ):
        reconstructed = sgt.wind(
            first,
            second,
            grid=grid,
            source=source,
            radius=radius,
        )
        _assert_close(reconstructed[0], eastward, dtype, tolerances=tolerances)
        _assert_close(reconstructed[1], northward, dtype, tolerances=tolerances)
