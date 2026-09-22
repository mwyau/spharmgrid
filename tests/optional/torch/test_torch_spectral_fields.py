# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reusable analyzed representations for the PyTorch backend."""

from __future__ import annotations

import numpy as np
import pytest

import spharmgrid as sg
import spharmgrid.torch as sgt
from tests.optional.torch.conftest import make_fields, torch


def test_scalar_spectral_field_reuses_native_coefficients_and_autograd(
    gl_grid: sg.Grid,
) -> None:
    field, _, _ = make_fields(gl_grid)
    field.requires_grad_()

    spectral = sgt.analyze(field, grid=gl_grid)
    assert isinstance(spectral, sgt.SpectralField)
    assert spectral._coefficients.device == field.device
    np.testing.assert_allclose(
        spectral.synthesize().detach().numpy(), field.detach().numpy(), atol=2.0e-12
    )
    np.testing.assert_allclose(
        spectral.laplacian().synthesize().detach().numpy(),
        sgt.laplacian(field, grid=gl_grid).detach().numpy(),
        atol=2.0e-12,
    )

    spectral.laplacian().synthesize().square().mean().backward()
    assert field.grad is not None
    assert bool(torch_isfinite(field.grad))


def test_vector_spectral_field_reuses_native_coefficients_and_autograd(
    gl_grid: sg.Grid,
) -> None:
    _, eastward, northward = make_fields(gl_grid)
    eastward.requires_grad_()
    northward.requires_grad_()

    spectral = sgt.analyze_vector(eastward, northward, grid=gl_grid)
    assert isinstance(spectral, sgt.SpectralVectorField)
    assert spectral._coefficients.device == eastward.device
    vorticity = spectral.vorticity().synthesize()
    expected = sgt.vorticity(eastward, northward, grid=gl_grid)
    np.testing.assert_allclose(vorticity.detach().numpy(), expected.detach().numpy())

    vorticity.square().mean().backward()
    assert eastward.grad is not None
    assert northward.grad is not None
    assert bool(torch_isfinite(eastward.grad))
    assert bool(torch_isfinite(northward.grad))


def test_restricted_scalar_domain_is_preserved_on_torch_regrid(
    gl_grid: sg.Grid,
    cc_grid: sg.Grid,
) -> None:
    field, _, _ = make_fields(gl_grid)
    spectral = sgt.analyze(field, "T2", grid=gl_grid)

    torch.testing.assert_close(
        spectral.synthesize(),
        sgt.filter(field, "T2", grid=gl_grid),
        rtol=0.0,
        atol=2.0e-12,
    )
    torch.testing.assert_close(
        spectral.regrid(cc_grid),
        sgt.regrid(field, cc_grid, "T2", source_grid=gl_grid),
        rtol=0.0,
        atol=2.0e-12,
    )
    torch.testing.assert_close(
        spectral.regrid(cc_grid, taper=0.1),
        sgt.regrid(field, cc_grid, "T2", source_grid=gl_grid, taper=0.1),
        rtol=0.0,
        atol=2.0e-12,
    )

    with pytest.raises(ValueError, match="exceeds"):
        spectral.regrid(cc_grid, "T3")


def test_unrestricted_domain_intersects_a_smaller_torch_target(
    gl_grid: sg.Grid,
) -> None:
    target = sg.gaussian_grid(4, 7)
    field, _, _ = make_fields(gl_grid)
    spectral = sgt.analyze(field, grid=gl_grid)

    torch.testing.assert_close(
        spectral.regrid(target),
        sgt.regrid(field, target, source_grid=gl_grid),
        rtol=0.0,
        atol=2.0e-12,
    )


def test_restricted_vector_domain_is_preserved_on_torch_regrid(
    gl_grid: sg.Grid,
    cc_grid: sg.Grid,
) -> None:
    _, eastward, northward = make_fields(gl_grid)
    spectral = sgt.analyze_vector(eastward, northward, "T2", grid=gl_grid)

    result_u, result_v = spectral.regrid(cc_grid)
    expected_u, expected_v = sgt.regrid_vector(
        eastward,
        northward,
        cc_grid,
        "T2",
        source_grid=gl_grid,
    )
    torch.testing.assert_close(result_u, expected_u, rtol=0.0, atol=2.0e-12)
    torch.testing.assert_close(result_v, expected_v, rtol=0.0, atol=2.0e-12)

    result_u, result_v = spectral.regrid(cc_grid, taper=0.1)
    expected_u, expected_v = sgt.regrid_vector(
        eastward,
        northward,
        cc_grid,
        "T2",
        source_grid=gl_grid,
        taper=0.1,
    )
    torch.testing.assert_close(result_u, expected_u, rtol=0.0, atol=2.0e-12)
    torch.testing.assert_close(result_v, expected_v, rtol=0.0, atol=2.0e-12)

    with pytest.raises(ValueError, match="exceeds"):
        spectral.regrid(cc_grid, "T3")


def torch_isfinite(value: torch.Tensor) -> torch.Tensor:
    """Return an all-finite check without adding a package-level API."""
    return torch.isfinite(value).all()
