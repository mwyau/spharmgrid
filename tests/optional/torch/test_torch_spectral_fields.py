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


def test_torch_reusable_filters_skip_true_no_ops(gl_grid: sg.Grid) -> None:
    field, eastward, northward = make_fields(gl_grid)
    scalar = sgt.analyze(field, grid=gl_grid)
    vector = sgt.analyze_vector(eastward, northward, grid=gl_grid)

    assert scalar.filter() is scalar
    assert scalar.filter(scalar.spec) is scalar
    assert vector.filter() is vector
    assert vector.filter(vector.spec) is vector


def test_torch_filter_updates_domain_and_rejects_expansion(
    gl_grid: sg.Grid,
    cc_grid: sg.Grid,
) -> None:
    field, _, _ = make_fields(gl_grid)
    full = sgt.analyze(field, grid=gl_grid)
    filtered = full.filter("T4")

    assert filtered.spec == sg.parse_spectral("T4")
    assert full.spec != filtered.spec
    assert filtered.filter("T4") is filtered
    assert filtered.filter("T2").spec == sg.parse_spectral("T2")
    with pytest.raises(ValueError, match="exceeds|cannot be restored"):
        filtered.filter("T5")
    with pytest.raises(ValueError, match="exceeds|cannot be restored"):
        filtered.regrid(cc_grid, "T5")
    assert filtered._coefficients.shape[-2:] == (5, 5)


def test_torch_cumulative_tapers_use_the_current_domain(
    gl_grid: sg.Grid,
    cc_grid: sg.Grid,
) -> None:
    field, _, _ = make_fields(gl_grid)

    actual = (
        sgt.analyze(field, grid=gl_grid)
        .filter("T4", taper=0.2)
        .filter("T4", taper=0.1)
        .synthesize()
    )
    expected = sgt.filter(field, "T4", grid=gl_grid, taper=0.02)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=2.0e-12)

    actual = sgt.analyze(field, grid=gl_grid).filter("T4").regrid(cc_grid, taper=0.1)
    expected = sgt.regrid(
        field,
        cc_grid,
        "T4",
        source_grid=gl_grid,
        taper=0.1,
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=2.0e-12)


def test_torch_filtered_vector_domain_propagates_and_preserves_autograd(
    gl_grid: sg.Grid,
) -> None:
    _, eastward, northward = make_fields(gl_grid)
    eastward.requires_grad_()
    northward.requires_grad_()
    spectral = sgt.analyze_vector(eastward, northward, grid=gl_grid).filter("T4")

    assert spectral.spec == sg.parse_spectral("T4")
    for derived in (
        spectral.vorticity(),
        spectral.divergence(),
        spectral.streamfunction(),
        spectral.velocity_potential(),
        spectral.divergent(),
        spectral.rotational(),
    ):
        assert derived.spec == spectral.spec
        derived.synthesize()

    spectral.laplacian().synthesize()[0].square().mean().backward()
    assert eastward.grad is not None
    assert northward.grad is not None
    assert bool(torch_isfinite(eastward.grad))
    assert bool(torch_isfinite(northward.grad))


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
