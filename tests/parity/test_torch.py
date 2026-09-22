# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Numerical parity between the Torch and default DUCC execution paths."""

from __future__ import annotations

import importlib

import numpy as np
import pytest
import xarray as xr

import spharmgrid as sg

torch = pytest.importorskip("torch")
pytest.importorskip("torch_harmonics")
sgt = importlib.import_module("spharmgrid.torch")

pytestmark = pytest.mark.parity


_TORCH_NUMERICAL_EXPORTS = {
    "filter",
    "regrid",
    "regrid_vector",
    "gradient",
    "inverse_gradient",
    "laplacian",
    "inverse_laplacian",
    "vector_laplacian",
    "inverse_vector_laplacian",
    "vorticity",
    "divergence",
    "kinematics",
    "streamfunction",
    "velocity_potential",
    "potentials",
    "helmholtz",
    "rotational_wind",
    "divergent_wind",
    "wind",
}
_TORCH_REUSABLE_EXPORTS = {
    "SpectralField",
    "SpectralVectorField",
    "analyze",
    "analyze_vector",
}


def _tolerances(dtype: torch.dtype, family: str) -> tuple[float, float]:
    """Return measured absolute-error floors with a small stability margin."""
    if dtype == torch.float64:
        atols = {
            "scalar_map": 1.0e-14,  # measured max 6.7e-15
            "vector_regrid": 1.0e-7,  # measured max 6.6e-8
            "gradient": 1.0e-20,  # measured max 1.6e-21
            "inverse_gradient": 2.0e-8,  # measured max 9.0e-9
            "laplacian": 3.0e-27,  # measured max 1.4e-27
            "inverse_laplacian": 5.0e-2,  # measured max 3.4e-2
            "vector_laplacian": 1.0e-19,  # measured max 4.3e-20
            "inverse_vector_laplacian": 5.0e5,  # measured max 3.7e5
            "kinematics": 1.0e-13,  # measured max 5.2e-14
            "potential": 2.0e-1,  # measured max 1.3e-1
            "wind": 2.0e-14,  # measured max 1.1e-14
            "helmholtz": 1.0e-7,  # measured max 6.8e-8
        }
    else:
        atols = {
            "scalar_map": 1.0e-6,  # measured max 4.8e-7
            "vector_regrid": 1.0e-6,  # measured max 7.5e-7
            "gradient": 2.0e-13,  # measured max 7.1e-14
            "inverse_gradient": 2.0e-7,  # measured max 9.7e-8
            "laplacian": 1.0e-19,  # measured max 7.2e-20
            "inverse_laplacian": 5.0e6,  # measured max 3.8e6
            "vector_laplacian": 5.0e-19,  # measured max 3.4e-19
            "inverse_vector_laplacian": 1.0e7,  # measured max 7.4e6
            "kinematics": 1.0e-12,  # measured max 4.8e-13
            "potential": 4.0,  # measured max 2.9e0
            "wind": 1.5e-6,  # measured max 9.1e-7
            "helmholtz": 1.0e-6,  # measured max 5.7e-7
        }
    return 0.0, atols[family]


def _assert_close(
    actual: torch.Tensor,
    expected: xr.DataArray | np.ndarray,
    dtype: torch.dtype,
    family: str,
) -> None:
    rtol, atol = _tolerances(dtype, family)
    expected_values = (
        expected.values if isinstance(expected, xr.DataArray) else expected
    )
    np.testing.assert_allclose(
        actual.detach().cpu().numpy(),
        expected_values,
        rtol=rtol,
        atol=atol,
    )


def _as_xarray(
    field: torch.Tensor,
    grid: sg.Grid,
    name: str = "field",
) -> xr.DataArray:
    """Create the Xarray reference input at the comparison boundary."""
    return xr.DataArray(
        field.detach().cpu().numpy().copy(),
        dims=("lat", "lon"),
        coords={"lat": grid.latitude, "lon": grid.longitude},
        name=name,
    )


def _as_tensor(field: xr.DataArray, dtype: torch.dtype) -> torch.Tensor:
    """Create a Torch input from an explicitly compared Xarray field."""
    return torch.as_tensor(field.values.copy(), dtype=dtype)


def _matching_source(
    field: xr.DataArray,
    grid: sg.Grid,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, xr.DataArray]:
    """Return identical dtype-specific source values for both backends."""
    tensor = _as_tensor(field, dtype)
    name = field.name if isinstance(field.name, str) else "field"
    return tensor, _as_xarray(tensor, grid, name)


def _scalar_values(grid: sg.Grid) -> np.ndarray:
    """Return deterministic scalar content through degree four."""
    latitude = np.deg2rad(grid.latitude)[:, None]
    longitude = np.deg2rad(grid.longitude)[None, :]
    sine = np.sin(latitude)
    cosine = np.cos(latitude)
    degree_two_zonal = (3.0 * sine**2 - 1.0) / 2.0
    degree_three_zonal = (5.0 * sine**3 - 3.0 * sine) / 2.0
    values = (
        0.8
        + 0.35 * sine
        + 0.25 * degree_two_zonal
        + 0.20 * cosine * np.cos(longitude)
        + 0.17 * sine * cosine * np.sin(longitude)
        + 0.13 * cosine**2 * np.cos(2.0 * longitude)
        + 0.11 * sine * cosine**2 * np.sin(2.0 * longitude)
        + 0.07 * cosine**3 * np.cos(3.0 * longitude)
        + 0.05 * degree_three_zonal
        + 0.03 * cosine**4 * np.sin(4.0 * longitude)
    )
    return np.broadcast_to(values, (grid.nlat, grid.nlon)).copy()


def _vector_values(grid: sg.Grid) -> tuple[np.ndarray, np.ndarray]:
    """Return a mixed divergent/rotational wind with degree-one and -two modes."""
    latitude = np.deg2rad(grid.latitude)[:, None]
    longitude = np.deg2rad(grid.longitude)[None, :]
    sine = np.sin(latitude)
    cosine = np.cos(latitude)

    # These are the physical gradient and rotated-gradient components of two
    # non-axisymmetric scalar potentials, with the common radius factor removed.
    divergent_eastward = -1.4 * np.sin(longitude) + sine * np.cos(2.0 * longitude)
    divergent_northward = -1.4 * sine * np.cos(longitude) + 0.5 * (
        cosine**2 - sine**2
    ) * np.sin(2.0 * longitude)
    rotational_eastward = 1.2 * sine * np.sin(longitude) + 0.8 * sine * cosine * np.cos(
        2.0 * longitude
    )
    rotational_northward = -1.2 * np.cos(longitude) - 0.8 * cosine * np.sin(
        2.0 * longitude
    )
    eastward = np.broadcast_to(
        divergent_eastward + rotational_eastward,
        (grid.nlat, grid.nlon),
    ).copy()
    northward = np.broadcast_to(
        divergent_northward + rotational_northward,
        (grid.nlat, grid.nlon),
    ).copy()
    return eastward, northward


@pytest.fixture(scope="module")
def gl_grid() -> sg.Grid:
    return sg.gaussian_grid(8, 18, latitude_order="descending", lon0=37.0)


@pytest.fixture(scope="module")
def gl_target_grid() -> sg.Grid:
    return sg.gaussian_grid(10, 20, latitude_order="ascending", lon0=-75.0)


@pytest.fixture(scope="module")
def cc_grid() -> sg.Grid:
    return sg.clenshaw_curtis_grid(17, 36, latitude_order="ascending", lon0=-75.0)


@pytest.fixture(scope="module")
def cc_target_grid() -> sg.Grid:
    return sg.clenshaw_curtis_grid(
        19,
        40,
        latitude_order="descending",
        lon0=123.0,
    )


def test_torch_exports_have_complete_parity_inventory() -> None:
    assert set(sgt.__all__) == _TORCH_NUMERICAL_EXPORTS | _TORCH_REUSABLE_EXPORTS | {
        "nn"
    }


def test_torch_reusable_spectral_fields_match_ducc(
    gl_grid: sg.Grid,
) -> None:
    field = torch.as_tensor(_scalar_values(gl_grid), dtype=torch.float64)
    reference = _as_xarray(field, gl_grid)
    spectral = sgt.analyze(field, grid=gl_grid)
    _assert_close(spectral.synthesize(), reference, torch.float64, "scalar_map")
    _assert_close(
        spectral.laplacian().synthesize(),
        sg.laplacian(reference),
        torch.float64,
        "laplacian",
    )

    eastward_values, northward_values = _vector_values(gl_grid)
    eastward = torch.as_tensor(eastward_values, dtype=torch.float64)
    northward = torch.as_tensor(northward_values, dtype=torch.float64)
    reference_u = _as_xarray(eastward, gl_grid, "u")
    reference_v = _as_xarray(northward, gl_grid, "v")
    vector = sgt.analyze_vector(eastward, northward, grid=gl_grid)
    _assert_close(
        vector.vorticity().synthesize(),
        sg.vorticity(reference_u, reference_v),
        torch.float64,
        "kinematics",
    )
    _assert_close(
        vector.divergence().synthesize(),
        sg.divergence(reference_u, reference_v),
        torch.float64,
        "kinematics",
    )
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_gl_scalar_spectral_parity(
    dtype: torch.dtype,
    gl_grid: sg.Grid,
    gl_target_grid: sg.Grid,
    cc_grid: sg.Grid,
) -> None:
    field = torch.as_tensor(_scalar_values(gl_grid), dtype=dtype)
    xarray_field = _as_xarray(field, gl_grid)

    expected_hard = sg.filter(xarray_field, "T2")
    actual_hard = sgt.filter(field, "T2", grid=gl_grid)
    _assert_close(actual_hard, expected_hard, dtype, "scalar_map")

    expected_band = sg.filter(xarray_field, "T1-3")
    actual_band = sgt.filter(field, "T1-3", grid=gl_grid)
    _assert_close(actual_band, expected_band, dtype, "scalar_map")

    expected_taper = sg.filter(xarray_field, "T3", taper=0.1)
    actual_taper = sgt.filter(field, "T3", taper=0.1, grid=gl_grid)
    _assert_close(actual_taper, expected_taper, dtype, "scalar_map")

    assert np.max(np.abs(expected_hard.values - xarray_field.values)) > 1.0e-2
    assert np.max(np.abs(expected_band.values - xarray_field.values)) > 1.0e-2

    expected_gl = sg.regrid(xarray_field, gl_target_grid, "T4")
    actual_gl = sgt.regrid(
        field,
        gl_target_grid,
        "T4",
        source_grid=gl_grid,
    )
    _assert_close(actual_gl, expected_gl, dtype, "scalar_map")

    expected_cc = sg.regrid(xarray_field, cc_grid, "T4")
    actual_cc = sgt.regrid(field, cc_grid, "T4", source_grid=gl_grid)
    _assert_close(actual_cc, expected_cc, dtype, "scalar_map")


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_gl_vector_regrid_parity(
    dtype: torch.dtype,
    gl_grid: sg.Grid,
    gl_target_grid: sg.Grid,
    cc_grid: sg.Grid,
) -> None:
    eastward_values, northward_values = _vector_values(gl_grid)
    eastward = torch.as_tensor(eastward_values, dtype=dtype)
    northward = torch.as_tensor(northward_values, dtype=dtype)
    xarray_eastward = _as_xarray(eastward, gl_grid, "u")
    xarray_northward = _as_xarray(northward, gl_grid, "v")

    for target_grid in (gl_target_grid, cc_grid):
        expected = sg.regrid_vector(
            xarray_eastward,
            xarray_northward,
            target_grid,
            "T4",
        )
        actual = sgt.regrid_vector(
            eastward,
            northward,
            target_grid,
            "T4",
            source_grid=gl_grid,
        )
        _assert_close(actual[0], expected.u, dtype, "vector_regrid")
        _assert_close(actual[1], expected.v, dtype, "vector_regrid")


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_gl_scalar_differential_parity(
    dtype: torch.dtype,
    gl_grid: sg.Grid,
) -> None:
    field = torch.as_tensor(_scalar_values(gl_grid), dtype=dtype)
    xarray_field = _as_xarray(field, gl_grid)

    expected_gradient = sg.gradient(xarray_field)
    actual_gradient = sgt.gradient(field, grid=gl_grid)
    _assert_close(
        actual_gradient[0],
        expected_gradient.gradient_eastward,
        dtype,
        "gradient",
    )
    _assert_close(
        actual_gradient[1],
        expected_gradient.gradient_northward,
        dtype,
        "gradient",
    )

    gradient_eastward, xarray_gradient_eastward = _matching_source(
        expected_gradient.gradient_eastward,
        gl_grid,
        dtype,
    )
    gradient_northward, xarray_gradient_northward = _matching_source(
        expected_gradient.gradient_northward,
        gl_grid,
        dtype,
    )
    expected_inverse_gradient = sg.inverse_gradient(
        xarray_gradient_eastward,
        xarray_gradient_northward,
    )
    actual_inverse_gradient = sgt.inverse_gradient(
        gradient_eastward,
        gradient_northward,
        grid=gl_grid,
    )
    _assert_close(
        actual_inverse_gradient,
        expected_inverse_gradient,
        dtype,
        "inverse_gradient",
    )

    expected_laplacian = sg.laplacian(xarray_field)
    actual_laplacian = sgt.laplacian(field, grid=gl_grid)
    _assert_close(actual_laplacian, expected_laplacian, dtype, "laplacian")

    expected_inverse_laplacian = sg.inverse_laplacian(xarray_field)
    actual_inverse_laplacian = sgt.inverse_laplacian(field, grid=gl_grid)
    _assert_close(
        actual_inverse_laplacian,
        expected_inverse_laplacian,
        dtype,
        "inverse_laplacian",
    )


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_gl_vector_operator_parity_and_source_branches(
    dtype: torch.dtype,
    gl_grid: sg.Grid,
) -> None:
    eastward_values, northward_values = _vector_values(gl_grid)
    eastward = torch.as_tensor(eastward_values, dtype=dtype)
    northward = torch.as_tensor(northward_values, dtype=dtype)
    xarray_eastward = _as_xarray(eastward, gl_grid, "u")
    xarray_northward = _as_xarray(northward, gl_grid, "v")

    expected_vector_laplacian = sg.vector_laplacian(
        xarray_eastward,
        xarray_northward,
    )
    actual_vector_laplacian = sgt.vector_laplacian(
        eastward,
        northward,
        grid=gl_grid,
    )
    _assert_close(
        actual_vector_laplacian[0],
        expected_vector_laplacian.u,
        dtype,
        "vector_laplacian",
    )
    _assert_close(
        actual_vector_laplacian[1],
        expected_vector_laplacian.v,
        dtype,
        "vector_laplacian",
    )

    expected_inverse_vector_laplacian = sg.inverse_vector_laplacian(
        xarray_eastward,
        xarray_northward,
    )
    actual_inverse_vector_laplacian = sgt.inverse_vector_laplacian(
        eastward,
        northward,
        grid=gl_grid,
    )
    _assert_close(
        actual_inverse_vector_laplacian[0],
        expected_inverse_vector_laplacian.u,
        dtype,
        "inverse_vector_laplacian",
    )
    _assert_close(
        actual_inverse_vector_laplacian[1],
        expected_inverse_vector_laplacian.v,
        dtype,
        "inverse_vector_laplacian",
    )

    expected_vorticity = sg.vorticity(xarray_eastward, xarray_northward)
    actual_vorticity = sgt.vorticity(eastward, northward, grid=gl_grid)
    _assert_close(actual_vorticity, expected_vorticity, dtype, "kinematics")

    expected_divergence = sg.divergence(xarray_eastward, xarray_northward)
    actual_divergence = sgt.divergence(eastward, northward, grid=gl_grid)
    _assert_close(actual_divergence, expected_divergence, dtype, "kinematics")

    expected_kinematics = sg.kinematics(xarray_eastward, xarray_northward)
    actual_kinematics = sgt.kinematics(eastward, northward, grid=gl_grid)
    _assert_close(actual_kinematics[0], expected_kinematics.vo, dtype, "kinematics")
    _assert_close(actual_kinematics[1], expected_kinematics.d, dtype, "kinematics")

    expected_streamfunction = sg.streamfunction(xarray_eastward, xarray_northward)
    actual_streamfunction = sgt.streamfunction(eastward, northward, grid=gl_grid)
    _assert_close(actual_streamfunction, expected_streamfunction, dtype, "potential")

    expected_velocity_potential = sg.velocity_potential(
        xarray_eastward,
        xarray_northward,
    )
    actual_velocity_potential = sgt.velocity_potential(
        eastward,
        northward,
        grid=gl_grid,
    )
    _assert_close(
        actual_velocity_potential,
        expected_velocity_potential,
        dtype,
        "potential",
    )

    expected_potentials = sg.potentials(xarray_eastward, xarray_northward)
    actual_potentials = sgt.potentials(eastward, northward, grid=gl_grid)
    _assert_close(actual_potentials[0], expected_potentials.strf, dtype, "potential")
    _assert_close(actual_potentials[1], expected_potentials.vp, dtype, "potential")

    expected_helmholtz = sg.helmholtz(xarray_eastward, xarray_northward)
    actual_helmholtz = sgt.helmholtz(eastward, northward, grid=gl_grid)
    for actual, name in zip(
        actual_helmholtz,
        ("u_divergent", "v_divergent", "u_rotational", "v_rotational"),
        strict=True,
    ):
        _assert_close(actual, expected_helmholtz[name], dtype, "helmholtz")

    reference_sources = {
        "vorticity": expected_vorticity,
        "streamfunction": expected_streamfunction,
        "divergence": expected_divergence,
        "velocity_potential": expected_velocity_potential,
    }
    for source in ("vorticity", "streamfunction"):
        source_tensor, source_xarray = _matching_source(
            reference_sources[source],
            gl_grid,
            dtype,
        )
        expected = sg.rotational_wind(source_xarray, source=source)
        actual = sgt.rotational_wind(
            source_tensor,
            grid=gl_grid,
            source=source,
        )
        _assert_close(actual[0], expected.u_rotational, dtype, "wind")
        _assert_close(actual[1], expected.v_rotational, dtype, "wind")

    for source in ("divergence", "velocity_potential"):
        source_tensor, source_xarray = _matching_source(
            reference_sources[source],
            gl_grid,
            dtype,
        )
        expected = sg.divergent_wind(source_xarray, source=source)
        actual = sgt.divergent_wind(
            source_tensor,
            grid=gl_grid,
            source=source,
        )
        _assert_close(actual[0], expected.u_divergent, dtype, "wind")
        _assert_close(actual[1], expected.v_divergent, dtype, "wind")

    for source, names in (
        ("vorticity_divergence", ("vorticity", "divergence")),
        ("potentials", ("streamfunction", "velocity_potential")),
    ):
        first_tensor, first_xarray = _matching_source(
            reference_sources[names[0]],
            gl_grid,
            dtype,
        )
        second_tensor, second_xarray = _matching_source(
            reference_sources[names[1]],
            gl_grid,
            dtype,
        )
        expected = sg.wind(
            first_xarray,
            second_xarray,
            source=source,
        )
        actual = sgt.wind(
            first_tensor,
            second_tensor,
            grid=gl_grid,
            source=source,
        )
        _assert_close(actual[0], expected.u, dtype, "wind")
        _assert_close(actual[1], expected.v, dtype, "wind")


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_cc_triangular_band_parity(
    dtype: torch.dtype,
    cc_grid: sg.Grid,
    cc_target_grid: sg.Grid,
    gl_grid: sg.Grid,
) -> None:
    scalar = torch.as_tensor(_scalar_values(cc_grid), dtype=dtype)
    xarray_scalar = _as_xarray(scalar, cc_grid)
    actual_filter = sgt.filter(scalar, "T4", grid=cc_grid)
    expected_filter = sg.filter(xarray_scalar, "T4")
    _assert_close(actual_filter, expected_filter, dtype, "scalar_map")

    for target_grid in (cc_target_grid, gl_grid):
        actual_scalar = sgt.regrid(
            scalar,
            target_grid,
            "T4",
            source_grid=cc_grid,
        )
        expected_scalar = sg.regrid(xarray_scalar, target_grid, "T4")
        _assert_close(actual_scalar, expected_scalar, dtype, "scalar_map")

    eastward_values, northward_values = _vector_values(cc_grid)
    eastward = torch.as_tensor(eastward_values, dtype=dtype)
    northward = torch.as_tensor(northward_values, dtype=dtype)
    xarray_eastward = _as_xarray(eastward, cc_grid, "u")
    xarray_northward = _as_xarray(northward, cc_grid, "v")
    for target_grid in (cc_target_grid, gl_grid):
        actual = sgt.regrid_vector(
            eastward,
            northward,
            target_grid,
            "T4",
            source_grid=cc_grid,
        )
        expected = sg.regrid_vector(
            xarray_eastward,
            xarray_northward,
            target_grid,
            "T4",
        )
        _assert_close(actual[0], expected.u, dtype, "vector_regrid")
        _assert_close(actual[1], expected.v, dtype, "vector_regrid")
