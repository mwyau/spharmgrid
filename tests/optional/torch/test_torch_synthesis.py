# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the native torch-harmonics inverse transforms."""

from __future__ import annotations

import pytest
import torch_harmonics

import spharmgrid as sg
from spharmgrid._transform import TransformSpec
from spharmgrid.grids import grid_layout
from spharmgrid.torch._backend import _TorchTransform
from tests.optional.torch.conftest import torch


def _grid(kind: str) -> sg.Grid:
    if kind == "cc":
        return sg.clenshaw_curtis_grid(9, 16, lon0=0.0)
    return sg.gaussian_grid(9, 17, lon0=0.0)


def _restore(values: torch.Tensor, grid: sg.Grid) -> torch.Tensor:
    layout = grid_layout(grid)
    values = values.index_select(
        -2,
        torch.as_tensor(layout.latitude.restore_indices, dtype=torch.long),
    )
    return values.index_select(
        -1,
        torch.as_tensor(layout.longitude.restore_indices, dtype=torch.long),
    )


@pytest.mark.parametrize("grid_kind", ["cc", "gl"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_scalar_synthesis_uses_native_torch_harmonics(
    grid_kind: str,
    dtype: torch.dtype,
) -> None:
    grid = _grid(grid_kind)
    spec = TransformSpec(0, 7, 7, "triangular")
    state = _TorchTransform(grid, grid, spec, scalar_synthesis=True).to(dtype=dtype)
    synthesis = state._scalar_synthesis
    assert isinstance(synthesis, torch_harmonics.InverseRealSHT)

    complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    coefficients = torch.randn(2, 8, 8, dtype=complex_dtype)
    actual = state.scalar_synthesis(coefficients)
    expected = _restore(synthesis(coefficients), grid)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert actual.dtype == dtype


@pytest.mark.parametrize("grid_kind", ["cc", "gl"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_vector_synthesis_uses_native_torch_harmonics(
    grid_kind: str,
    dtype: torch.dtype,
) -> None:
    grid = _grid(grid_kind)
    spec = TransformSpec(0, 7, 7, "triangular")
    state = _TorchTransform(grid, grid, spec, vector_synthesis=True).to(dtype=dtype)
    synthesis = state._vector_synthesis
    assert isinstance(synthesis, torch_harmonics.InverseRealVectorSHT)

    complex_dtype = torch.complex128 if dtype is torch.float64 else torch.complex64
    coefficients = torch.randn(2, 2, 8, 8, dtype=complex_dtype)
    actual = state.vector_synthesis(coefficients)
    degrees = torch.arange(8, dtype=dtype).reshape(8, 1)
    scale = torch.sqrt(degrees * (degrees + 1.0))
    safe_scale = torch.where(scale > 0.0, scale, torch.ones_like(scale))
    native_coefficients = torch.stack(
        (
            torch.where(scale > 0.0, coefficients[:, 0] / safe_scale, 0.0),
            torch.where(scale > 0.0, -coefficients[:, 1] / safe_scale, 0.0),
        ),
        dim=-3,
    )
    native_values = synthesis(native_coefficients)
    expected = (
        _restore(native_values.select(-3, 1), grid),
        _restore(-native_values.select(-3, 0), grid),
    )
    for actual_component, expected_component in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            actual_component, expected_component, rtol=0.0, atol=0.0
        )
        assert actual_component.dtype == dtype
