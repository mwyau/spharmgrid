"""Optional Torch differential and wind-operator tests."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest

import spharmgrid as sg

torch = pytest.importorskip("torch")
sgt = pytest.importorskip("spharmgrid.torch")  # noqa: E402

from tests.optional.torch.conftest import as_xarray, make_fields  # noqa: E402


def _assert_close(
    actual: torch.Tensor,
    expected: np.ndarray,
    dtype: torch.dtype,
    *,
    rtol: float | None = None,
    atol: float | None = None,
) -> None:
    if rtol is None:
        rtol = 3.0e-11 if dtype == torch.float64 else 3.0e-5
    if atol is None:
        atol = 3.0e-10 if dtype == torch.float64 else 3.0e-4
    np.testing.assert_allclose(
        actual.detach().cpu().numpy(),
        expected,
        rtol=rtol,
        atol=atol,
    )


def _physical_tolerances(
    dtype: torch.dtype,
    *,
    vector: bool,
) -> tuple[float, float]:
    """Set tolerances from measured transform error and output scale."""
    if vector:
        return (1.0e-9, 1.0e5) if dtype == torch.float64 else (5.0e-7, 1.0e6)
    return (1.0e-9, 0.1) if dtype == torch.float64 else (5.0e-7, 5.0)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_scalar_operators_match_cpu(gl_grid: sg.Grid, dtype: torch.dtype) -> None:
    field, _, _ = make_fields(gl_grid, dtype)
    xarray_field = as_xarray(field, gl_grid)

    expected_gradient = sg.gradient(xarray_field)
    actual_gradient = sgt.gradient(field, grid=gl_grid)
    _assert_close(actual_gradient[0], expected_gradient.gradient_eastward.values, dtype)
    _assert_close(
        actual_gradient[1], expected_gradient.gradient_northward.values, dtype
    )

    expected_inverse_gradient = sg.inverse_gradient(
        expected_gradient.gradient_eastward,
        expected_gradient.gradient_northward,
    )
    actual_inverse_gradient = sgt.inverse_gradient(
        *actual_gradient,
        grid=gl_grid,
    )
    _assert_close(
        actual_inverse_gradient,
        expected_inverse_gradient.values,
        dtype,
        rtol=3.0e-8 if dtype == torch.float64 else None,
        atol=3.0e-8 if dtype == torch.float64 else None,
    )

    for operation, expected in (
        (sgt.laplacian(field, grid=gl_grid), sg.laplacian(xarray_field).values),
        (
            sgt.inverse_laplacian(field, grid=gl_grid),
            sg.inverse_laplacian(xarray_field).values,
        ),
    ):
        _assert_close(operation, expected, dtype)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_vector_operators_match_cpu(gl_grid: sg.Grid, dtype: torch.dtype) -> None:
    _, eastward, northward = make_fields(gl_grid, dtype)
    xarray_eastward = as_xarray(eastward, gl_grid, "u")
    xarray_northward = as_xarray(northward, gl_grid, "v")

    actual_vorticity = sgt.vorticity(eastward, northward, grid=gl_grid)
    actual_divergence = sgt.divergence(eastward, northward, grid=gl_grid)
    expected_kinematics = sg.kinematics(xarray_eastward, xarray_northward)
    _assert_close(actual_vorticity, expected_kinematics.vo.values, dtype)
    _assert_close(actual_divergence, expected_kinematics.d.values, dtype)

    actual_kinematics = sgt.kinematics(eastward, northward, grid=gl_grid)
    _assert_close(actual_kinematics[0], expected_kinematics.vo.values, dtype)
    _assert_close(actual_kinematics[1], expected_kinematics.d.values, dtype)

    actual_potentials = sgt.potentials(eastward, northward, grid=gl_grid)
    expected_potentials = sg.potentials(xarray_eastward, xarray_northward)
    potential_rtol, potential_atol = _physical_tolerances(dtype, vector=False)
    _assert_close(
        actual_potentials[0],
        expected_potentials.strf.values,
        dtype,
        rtol=potential_rtol,
        atol=potential_atol,
    )
    _assert_close(
        actual_potentials[1],
        expected_potentials.vp.values,
        dtype,
        rtol=potential_rtol,
        atol=potential_atol,
    )
    _assert_close(
        sgt.streamfunction(eastward, northward, grid=gl_grid),
        expected_potentials.strf.values,
        dtype,
        rtol=potential_rtol,
        atol=potential_atol,
    )
    _assert_close(
        sgt.velocity_potential(eastward, northward, grid=gl_grid),
        expected_potentials.vp.values,
        dtype,
        rtol=potential_rtol,
        atol=potential_atol,
    )

    expected_vector_laplacian = sg.vector_laplacian(
        xarray_eastward,
        xarray_northward,
    )
    actual_vector_laplacian = sgt.vector_laplacian(
        eastward,
        northward,
        grid=gl_grid,
    )
    _assert_close(actual_vector_laplacian[0], expected_vector_laplacian.u.values, dtype)
    _assert_close(actual_vector_laplacian[1], expected_vector_laplacian.v.values, dtype)

    expected_inverse_vector_laplacian = sg.inverse_vector_laplacian(
        xarray_eastward,
        xarray_northward,
    )
    inverse_vector_rtol, inverse_vector_atol = _physical_tolerances(
        dtype,
        vector=True,
    )
    actual_inverse_vector_laplacian = sgt.inverse_vector_laplacian(
        eastward,
        northward,
        grid=gl_grid,
    )
    _assert_close(
        actual_inverse_vector_laplacian[0],
        expected_inverse_vector_laplacian.u.values,
        dtype,
        rtol=inverse_vector_rtol,
        atol=inverse_vector_atol,
    )
    _assert_close(
        actual_inverse_vector_laplacian[1],
        expected_inverse_vector_laplacian.v.values,
        dtype,
        rtol=inverse_vector_rtol,
        atol=inverse_vector_atol,
    )

    expected_helmholtz = sg.helmholtz(xarray_eastward, xarray_northward)
    actual_helmholtz = sgt.helmholtz(eastward, northward, grid=gl_grid)
    for actual, name in zip(
        actual_helmholtz,
        ("u_divergent", "v_divergent", "u_rotational", "v_rotational"),
        strict=True,
    ):
        _assert_close(
            actual,
            expected_helmholtz[name].values,
            dtype,
            rtol=3.0e-8 if dtype == torch.float64 else None,
            atol=3.0e-8 if dtype == torch.float64 else None,
        )

    expected_rotational = sg.rotational_wind(
        expected_kinematics.vo,
        quantity="vorticity",
    )
    actual_rotational = sgt.rotational_wind(
        actual_vorticity,
        grid=gl_grid,
        quantity="vorticity",
    )
    _assert_close(
        actual_rotational[0],
        expected_rotational.u_rotational.values,
        dtype,
        rtol=3.0e-8 if dtype == torch.float64 else None,
        atol=3.0e-8 if dtype == torch.float64 else None,
    )
    _assert_close(
        actual_rotational[1],
        expected_rotational.v_rotational.values,
        dtype,
        rtol=3.0e-8 if dtype == torch.float64 else None,
        atol=3.0e-8 if dtype == torch.float64 else None,
    )

    expected_divergent = sg.divergent_wind(
        expected_kinematics.d,
        quantity="divergence",
    )
    actual_divergent = sgt.divergent_wind(
        actual_divergence,
        grid=gl_grid,
        quantity="divergence",
    )
    _assert_close(
        actual_divergent[0],
        expected_divergent.u_divergent.values,
        dtype,
        rtol=3.0e-8 if dtype == torch.float64 else None,
        atol=3.0e-8 if dtype == torch.float64 else None,
    )
    _assert_close(
        actual_divergent[1],
        expected_divergent.v_divergent.values,
        dtype,
        rtol=3.0e-8 if dtype == torch.float64 else None,
        atol=3.0e-8 if dtype == torch.float64 else None,
    )

    expected_wind = sg.wind(
        expected_kinematics.vo,
        expected_kinematics.d,
        source="vorticity_divergence",
    )
    actual_wind = sgt.wind(
        actual_vorticity,
        actual_divergence,
        grid=gl_grid,
        source="vorticity_divergence",
    )
    _assert_close(
        actual_wind[0],
        expected_wind.u.values,
        dtype,
        rtol=3.0e-8 if dtype == torch.float64 else None,
        atol=3.0e-8 if dtype == torch.float64 else None,
    )
    _assert_close(
        actual_wind[1],
        expected_wind.v.values,
        dtype,
        rtol=3.0e-8 if dtype == torch.float64 else None,
        atol=3.0e-8 if dtype == torch.float64 else None,
    )

    expected_potential_wind = sg.wind(
        expected_potentials.strf,
        expected_potentials.vp,
        source="potentials",
    )
    actual_potential_wind = sgt.wind(
        actual_potentials[0],
        actual_potentials[1],
        grid=gl_grid,
        source="potentials",
    )
    _assert_close(
        actual_potential_wind[0],
        expected_potential_wind.u.values,
        dtype,
        rtol=3.0e-8 if dtype == torch.float64 else None,
        atol=3.0e-8 if dtype == torch.float64 else None,
    )
    _assert_close(
        actual_potential_wind[1],
        expected_potential_wind.v.values,
        dtype,
        rtol=3.0e-8 if dtype == torch.float64 else None,
        atol=3.0e-8 if dtype == torch.float64 else None,
    )


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
        atol=5.0e-7,
    )
    _assert_close(
        velocity_potential,
        divergent_amplitude * radius * sine,
        torch.float64,
        atol=5.0e-7,
    )

    recovered_rotational = sgt.rotational_wind(
        vorticity,
        grid=grid,
        quantity="vorticity",
        radius=radius,
    )
    recovered_divergent = sgt.divergent_wind(
        divergence,
        grid=grid,
        quantity="divergence",
        radius=radius,
    )
    _assert_close(
        recovered_rotational[0], eastward.numpy(), torch.float64, atol=5.0e-14
    )
    _assert_close(
        recovered_rotational[1], np.zeros_like(cosine), torch.float64, atol=5.0e-20
    )
    _assert_close(
        recovered_divergent[0], np.zeros_like(cosine), torch.float64, atol=5.0e-20
    )
    _assert_close(
        recovered_divergent[1], northward.numpy(), torch.float64, atol=5.0e-14
    )

    recovered_wind = sgt.wind(
        vorticity,
        divergence,
        grid=grid,
        source="vorticity_divergence",
        radius=radius,
    )
    _assert_close(recovered_wind[0], eastward.numpy(), torch.float64, atol=5.0e-14)
    _assert_close(recovered_wind[1], northward.numpy(), torch.float64, atol=5.0e-14)


@pytest.mark.parametrize("latitude_order", ["ascending", "descending"])
def test_analytic_nonaxisymmetric_l1m1_vector_identities(
    latitude_order: Literal["ascending", "descending"],
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
    u = torch.as_tensor(eastward, dtype=torch.float64)
    v = torch.as_tensor(northward, dtype=torch.float64)
    scalar = torch.as_tensor(chi, dtype=torch.float64)

    actual_gradient = sgt.gradient(scalar, grid=grid, radius=radius)
    _assert_close(
        actual_gradient[0], divergent_u, torch.float64, rtol=2.0e-12, atol=2.0e-12
    )
    _assert_close(
        actual_gradient[1], divergent_v, torch.float64, rtol=2.0e-12, atol=2.0e-12
    )
    _assert_close(
        sgt.inverse_gradient(
            torch.as_tensor(divergent_u, dtype=torch.float64),
            torch.as_tensor(divergent_v, dtype=torch.float64),
            grid=grid,
            radius=radius,
        ),
        chi,
        torch.float64,
        rtol=2.0e-12,
        atol=2.0e-12,
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
        torch.float64,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    _assert_close(
        actual_divergence,
        -2.0 * chi / radius**2,
        torch.float64,
        rtol=2.0e-12,
        atol=2.0e-12,
    )

    actual_potentials = sgt.potentials(u, v, grid=grid, radius=radius)
    _assert_close(actual_potentials[0], psi, torch.float64, rtol=2.0e-12, atol=2.0e-12)
    _assert_close(actual_potentials[1], chi, torch.float64, rtol=2.0e-12, atol=2.0e-12)

    actual_helmholtz = sgt.helmholtz(u, v, grid=grid, radius=radius)
    for actual, expected in zip(
        actual_helmholtz,
        (divergent_u, divergent_v, rotational_u, rotational_v),
        strict=True,
    ):
        _assert_close(actual, expected, torch.float64, rtol=2.0e-12, atol=2.0e-12)

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
        _assert_close(
            reconstructed[0], eastward, torch.float64, rtol=2.0e-12, atol=2.0e-12
        )
        _assert_close(
            reconstructed[1], northward, torch.float64, rtol=2.0e-12, atol=2.0e-12
        )
