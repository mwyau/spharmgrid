# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""High-bandwidth Torch analysis tests on pole-including CC grids."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch_harmonics
import xarray as xr

import spharmgrid as sg
import spharmgrid.torch as sgt
import spharmgrid.torch.nn as sgnn
from spharmgrid._ducc import (
    alm_degrees,
    alm_orders,
    geometry_for,
    scalar_synthesis,
    vector_synthesis,
)
from spharmgrid._transform import TransformSpec
from spharmgrid.grids import grid_layout
from spharmgrid.torch._backend import _TorchTransform
from spharmgrid.torch._cc_sht import (
    _ExtendedCCRealSHT,
    _ExtendedCCRealVectorSHT,
    _fourier_shift_latitude,
    _fourier_shift_latitude_adjoint,
    _shift_phase,
)
from tests.optional.torch.conftest import torch

_HIGH_SPEC = TransformSpec(0, 71, 71, "triangular")
_HIGH_MODES = ((70, 0), (70, 1), (70, 70), (71, 0), (71, 1), (71, 71))


@pytest.fixture(scope="module")
def high_cc_grid() -> sg.Grid:
    return sg.clenshaw_curtis_grid(73, 144, latitude_order="descending")


def _mode_coefficients(degree: int, order: int) -> np.ndarray:
    degrees = alm_degrees(_HIGH_SPEC.lmax, _HIGH_SPEC.mmax)
    orders = alm_orders(_HIGH_SPEC.lmax, _HIGH_SPEC.mmax)
    index = np.flatnonzero((degrees == degree) & (orders == order))
    assert index.size == 1
    coefficients = np.zeros(degrees.size, dtype=np.complex128)
    coefficients[index[0]] = 1.0
    return coefficients


def _scalar_mode(grid: sg.Grid, degree: int, order: int) -> np.ndarray:
    layout = grid_layout(grid)
    values = scalar_synthesis(
        _mode_coefficients(degree, order)[None, :],
        spec=_HIGH_SPEC,
        geometry=geometry_for(grid),
        ntheta=grid.nlat,
        nphi=grid.nlon,
        phi0=layout.phi0_radians,
        nthreads=0,
    )
    return values[layout.latitude.restore_indices][:, layout.longitude.restore_indices]


def _vector_mode(
    grid: sg.Grid,
    degree: int,
    order: int,
    channel: int,
) -> tuple[np.ndarray, np.ndarray]:
    layout = grid_layout(grid)
    coefficients = _mode_coefficients(degree, order)
    zeros = np.zeros_like(coefficients)
    if channel == 0:
        eastward, northward = vector_synthesis(
            coefficients,
            zeros,
            spec=_HIGH_SPEC,
            geometry=geometry_for(grid),
            ntheta=grid.nlat,
            nphi=grid.nlon,
            phi0=layout.phi0_radians,
            nthreads=0,
        )
    else:
        eastward, northward = vector_synthesis(
            zeros,
            coefficients,
            spec=_HIGH_SPEC,
            geometry=geometry_for(grid),
            ntheta=grid.nlat,
            nphi=grid.nlon,
            phi0=layout.phi0_radians,
            nthreads=0,
        )
    latitude_indices = layout.latitude.restore_indices
    longitude_indices = layout.longitude.restore_indices
    return (
        eastward[latitude_indices][:, longitude_indices],
        northward[latitude_indices][:, longitude_indices],
    )


def _labeled_field(values: np.ndarray, grid: sg.Grid, name: str) -> xr.DataArray:
    return xr.DataArray(
        values,
        dims=("lat", "lon"),
        coords={"lat": grid.latitude, "lon": grid.longitude},
        name=name,
    )


def test_cc_shift_phase_uses_fft_nyquist_frequency() -> None:
    nlat = 9
    periodic_length = 2 * (nlat - 1)
    actual = _shift_phase(periodic_length)
    frequencies = torch.fft.fftfreq(periodic_length, dtype=torch.float64)
    expected = torch.stack(
        (
            torch.cos(math.pi * frequencies),
            torch.sin(math.pi * frequencies),
        ),
        dim=-1,
    )

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1.0e-15)
    torch.testing.assert_close(
        actual[periodic_length // 2],
        torch.tensor([0.0, -1.0], dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-15,
    )


def test_fourier_shift_latitude_adjoint_even_length() -> None:
    nlat = 9
    periodic_length = 2 * (nlat - 1)
    sample = torch.arange(
        2 * 3 * periodic_length,
        dtype=torch.float64,
    ).reshape(2, 3, periodic_length)
    x = torch.complex(torch.sin(sample / 7.0), torch.cos(sample / 11.0))
    y = torch.complex(torch.cos(sample / 5.0), torch.sin(sample / 13.0))
    phase_values = _shift_phase(periodic_length).to(
        device=x.device,
        dtype=x.real.dtype,
    )
    phase = torch.complex(phase_values[:, 0], phase_values[:, 1])

    lhs = torch.vdot(
        _fourier_shift_latitude(x, phase).reshape(-1),
        y.reshape(-1),
    )
    rhs = torch.vdot(
        x.reshape(-1),
        _fourier_shift_latitude_adjoint(y, phase).reshape(-1),
    )
    torch.testing.assert_close(lhs, rhs, rtol=1.0e-12, atol=1.0e-12)


def test_cc_analysis_activation_boundary() -> None:
    grid = sg.clenshaw_curtis_grid(17, 36)
    low = TransformSpec(0, 8, 8, "triangular")
    high = TransformSpec(0, 9, 9, "triangular")

    low_scalar = _TorchTransform(grid, grid, low, scalar_analysis=True)
    high_scalar = _TorchTransform(grid, grid, high, scalar_analysis=True)
    low_vector = _TorchTransform(grid, grid, low, vector_analysis=True)
    high_vector = _TorchTransform(grid, grid, high, vector_analysis=True)

    assert isinstance(low_scalar._scalar_analysis, torch_harmonics.RealSHT)
    assert isinstance(high_scalar._scalar_analysis, _ExtendedCCRealSHT)
    assert isinstance(low_vector._vector_analysis, torch_harmonics.RealVectorSHT)
    assert isinstance(high_vector._vector_analysis, _ExtendedCCRealVectorSHT)


def test_extended_cc_projection_matches_native_direct_band() -> None:
    grid = sg.clenshaw_curtis_grid(9, 16)
    latitude = torch.as_tensor(np.deg2rad(grid.latitude), dtype=torch.float64)[:, None]
    longitude = torch.as_tensor(np.deg2rad(grid.longitude), dtype=torch.float64)[
        None, :
    ]
    field = torch.sin(latitude) + torch.cos(latitude) * torch.cos(longitude)
    vector_field = torch.stack(
        (
            (torch.sin(latitude) * torch.cos(longitude)).expand_as(field),
            (-torch.sin(longitude)).expand_as(field),
        )
    )
    lmax = mmax = 5

    native_scalar = torch_harmonics.RealSHT(
        grid.nlat,
        grid.nlon,
        lmax=lmax,
        mmax=mmax,
        grid="equiangular",
        norm="ortho",
        csphase=True,
    )
    extended_scalar = _ExtendedCCRealSHT(
        grid.nlat,
        grid.nlon,
        lmax=lmax,
        mmax=mmax,
    )
    torch.testing.assert_close(
        extended_scalar(field),
        native_scalar(field),
        rtol=2.0e-13,
        atol=2.0e-14,
    )

    native_vector = torch_harmonics.RealVectorSHT(
        grid.nlat,
        grid.nlon,
        lmax=lmax,
        mmax=mmax,
        grid="equiangular",
        norm="ortho",
        csphase=True,
    )
    extended_vector = _ExtendedCCRealVectorSHT(
        grid.nlat,
        grid.nlon,
        lmax=lmax,
        mmax=mmax,
    )
    torch.testing.assert_close(
        extended_vector(vector_field),
        native_vector(vector_field),
        rtol=2.0e-13,
        atol=2.0e-14,
    )


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_high_cc_scalar_analysis_matches_ducc_modes(
    dtype: torch.dtype,
    high_cc_grid: sg.Grid,
) -> None:
    state = _TorchTransform(
        high_cc_grid,
        high_cc_grid,
        _HIGH_SPEC,
        scalar_analysis=True,
        dtype=dtype,
    )
    assert state._scalar_analysis is not None
    assert state._scalar_analysis._projection is not None
    assert state._scalar_analysis._projection.dtype == dtype
    tolerance = 4.0e-13 if dtype == torch.float64 else 4.0e-7
    for degree, order in _HIGH_MODES:
        actual = state.scalar_analysis(
            torch.as_tensor(_scalar_mode(high_cc_grid, degree, order), dtype=dtype)
        )
        expected = torch.zeros_like(actual)
        expected[degree, order] = 1.0 + 0.0j
        assert float((actual - expected).abs().max()) <= tolerance


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("channel", [0, 1])
def test_high_cc_vector_analysis_matches_ducc_e_and_b_modes(
    dtype: torch.dtype,
    channel: int,
    high_cc_grid: sg.Grid,
) -> None:
    state = _TorchTransform(
        high_cc_grid,
        high_cc_grid,
        _HIGH_SPEC,
        vector_analysis=True,
        dtype=dtype,
    )
    assert state._vector_analysis is not None
    assert state._vector_analysis._projection is not None
    assert state._vector_analysis._projection.dtype == dtype
    tolerance = 5.0e-8 if dtype == torch.float64 else 5.0e-7
    for degree, order in _HIGH_MODES:
        eastward, northward = _vector_mode(
            high_cc_grid,
            degree,
            order,
            channel,
        )
        actual = state.vector_analysis(
            torch.as_tensor(eastward, dtype=dtype),
            torch.as_tensor(northward, dtype=dtype),
        )
        expected = torch.zeros_like(actual)
        expected[channel, degree, order] = 1.0 + 0.0j
        assert float((actual - expected).abs().max()) <= tolerance


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_high_band_cc_public_operations_match_ducc(dtype: torch.dtype) -> None:
    source = sg.clenshaw_curtis_grid(
        73,
        144,
        latitude_order="ascending",
        lon0=37.0,
    )
    target = sg.clenshaw_curtis_grid(
        73,
        144,
        latitude_order="descending",
        lon0=-75.0,
    )
    scalar = _labeled_field(_scalar_mode(source, 71, 1), source, "field")
    scalar_tensor = torch.as_tensor(scalar.values, dtype=dtype)

    scalar_tolerance = 1.0e-12 if dtype == torch.float64 else 2.0e-6
    vector_tolerance = 1.0e-7 if dtype == torch.float64 else 4.0e-6
    scalar_results = (
        (
            sgt.filter(scalar_tensor, "T71", grid=source),
            sg.filter(scalar, "T71"),
        ),
        (
            sgt.regrid(scalar_tensor, target, "T71", source_grid=source),
            sg.regrid(scalar, target, "T71"),
        ),
    )
    for actual, expected in scalar_results:
        np.testing.assert_allclose(
            actual.detach().cpu().numpy(),
            expected.values,
            rtol=0.0,
            atol=scalar_tolerance,
        )

    eastward_values, northward_values = _vector_mode(source, 71, 71, 0)
    eastward = _labeled_field(eastward_values, source, "u")
    northward = _labeled_field(northward_values, source, "v")
    eastward_tensor = torch.as_tensor(eastward_values, dtype=dtype)
    northward_tensor = torch.as_tensor(northward_values, dtype=dtype)
    actual_vector = sgt.regrid_vector(
        eastward_tensor,
        northward_tensor,
        target,
        "T71",
        source_grid=source,
    )
    expected_vector = sg.regrid_vector(eastward, northward, target, "T71")
    for actual, expected in zip(
        actual_vector,
        (expected_vector.u, expected_vector.v),
        strict=True,
    ):
        np.testing.assert_allclose(
            actual.detach().cpu().numpy(),
            expected.values,
            rtol=0.0,
            atol=vector_tolerance,
        )

    operators = sgnn.SHTOperators(source).to(dtype=dtype)
    np.testing.assert_allclose(
        operators.filter(scalar_tensor).detach().cpu().numpy(),
        sg.filter(scalar, "T71").values,
        rtol=0.0,
        atol=scalar_tolerance,
    )
    actual_kinematics = operators.kinematics(eastward_tensor, northward_tensor)
    expected_kinematics = sg.kinematics(eastward, northward)
    for actual, expected in zip(
        actual_kinematics,
        (expected_kinematics.vo, expected_kinematics.d),
        strict=True,
    ):
        np.testing.assert_allclose(
            actual.detach().cpu().numpy(),
            expected.values,
            rtol=0.0,
            atol=scalar_tolerance,
        )


def test_high_cc_leading_dimensions_are_preserved() -> None:
    grid = sg.clenshaw_curtis_grid(9, 16)
    field = torch.linspace(
        -1.0,
        1.0,
        grid.nlat * grid.nlon,
        dtype=torch.float64,
    ).reshape(grid.nlat, grid.nlon)
    batched = field.reshape(1, 1, grid.nlat, grid.nlon).expand(2, 3, -1, -1)
    result = sgt.filter(batched, "T7", grid=grid)
    assert result.shape == batched.shape


def test_high_cc_scalar_analysis_passes_gradcheck() -> None:
    grid = sg.clenshaw_curtis_grid(9, 16)
    field = torch.linspace(
        -1.0,
        1.0,
        grid.nlat * grid.nlon,
        dtype=torch.float64,
        requires_grad=True,
    ).reshape(grid.nlat, grid.nlon)
    assert torch.autograd.gradcheck(
        lambda value: sgt.filter(value, "T7", grid=grid),
        (field,),
        eps=1.0e-6,
        atol=1.0e-7,
        rtol=1.0e-5,
    )


def test_high_cc_vector_analysis_passes_gradcheck() -> None:
    grid = sg.clenshaw_curtis_grid(9, 16)
    field = torch.linspace(
        -1.0,
        1.0,
        grid.nlat * grid.nlon,
        dtype=torch.float64,
        requires_grad=True,
    ).reshape(grid.nlat, grid.nlon)
    eastward = field.detach().clone().requires_grad_()
    northward = torch.flip(field.detach(), dims=(-2,)).requires_grad_()
    assert torch.autograd.gradcheck(
        lambda u, v: sgt.kinematics(u, v, grid=grid),
        (eastward, northward),
        eps=1.0e-6,
        atol=1.0e-7,
        rtol=1.0e-5,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_full_band_cc_operations_preserve_cuda_autograd() -> None:
    grid = sg.clenshaw_curtis_grid(73, 144)
    operators = sgnn.SHTOperators(grid).to(device="cuda", dtype=torch.float32)
    state_buffers = list(operators._state.buffers())
    floating_buffers = [
        buffer for buffer in state_buffers if buffer.is_floating_point()
    ]
    integer_buffers = [
        buffer for buffer in state_buffers if not buffer.is_floating_point()
    ]
    assert all(buffer.device.type == "cuda" for buffer in state_buffers)
    assert all(buffer.dtype == torch.float32 for buffer in floating_buffers)
    assert all(buffer.dtype == torch.long for buffer in integer_buffers)
    field = torch.randn(
        grid.nlat,
        grid.nlon,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    eastward = torch.randn_like(field, requires_grad=True)
    northward = torch.randn_like(field, requires_grad=True)

    filtered = operators.filter(field, "T71")
    vorticity, divergence = operators.kinematics(eastward, northward)
    loss = (
        filtered.square().mean()
        + vorticity.square().mean()
        + divergence.square().mean()
    )
    loss.backward()

    assert filtered.device == field.device
    assert filtered.dtype == field.dtype
    assert vorticity.device == field.device
    assert divergence.device == field.device
    assert field.grad is not None
    assert eastward.grad is not None
    assert northward.grad is not None
    assert field.grad.device == field.device
    assert bool(torch.isfinite(field.grad).all())
    assert bool(torch.isfinite(eastward.grad).all())
    assert bool(torch.isfinite(northward.grad).all())


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile unavailable")
def test_high_cc_filter_runs_under_torch_compile_eager() -> None:
    grid = sg.clenshaw_curtis_grid(9, 16)
    field = torch.linspace(
        -1.0,
        1.0,
        grid.nlat * grid.nlon,
        dtype=torch.float32,
    ).reshape(grid.nlat, grid.nlon)
    layer = sgnn.SHTFilter(grid, "T7").to(dtype=torch.float32)
    compiled = torch.compile(layer, backend="eager")
    torch.testing.assert_close(compiled(field), layer(field))
