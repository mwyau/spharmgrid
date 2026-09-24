# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reusable analyzed representations for the PyTorch backend."""

from __future__ import annotations

import numpy as np
import pytest

import spharmgrid as sg
import spharmgrid.torch as sgt
import spharmgrid.torch._backend as torch_backend
import spharmgrid.torch.nn as sgnn
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


def test_scalar_transform_state_omits_vector_directions(gl_grid: sg.Grid) -> None:
    layer = sgnn.SHTFilter(gl_grid, "T4")

    assert layer._state._scalar_analysis is not None
    assert layer._state._scalar_synthesis is not None
    assert layer._state._vector_analysis is None
    assert layer._state._vector_synthesis is None


def test_transform_state_allocates_only_requested_directions(
    gl_grid: sg.Grid,
) -> None:
    spec = sg.parse_spectral("T4")
    vector_analysis = torch_backend._TorchTransform(
        gl_grid,
        gl_grid,
        spec,
        vector_analysis=True,
    )
    assert vector_analysis._vector_analysis is not None
    assert vector_analysis._vector_synthesis is None
    assert vector_analysis._scalar_analysis is None
    assert vector_analysis._scalar_synthesis is None

    vector_synthesis = torch_backend._TorchTransform(
        gl_grid,
        gl_grid,
        spec,
        vector_synthesis=True,
    )
    assert vector_synthesis._vector_analysis is None
    assert vector_synthesis._vector_synthesis is not None
    assert vector_synthesis._scalar_analysis is None
    assert vector_synthesis._scalar_synthesis is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_native_projection_buffers_follow_state_dtype(
    gl_grid: sg.Grid,
    dtype: torch.dtype,
) -> None:
    state = torch_backend._TorchTransform(
        gl_grid,
        gl_grid,
        sg.parse_spectral("T4"),
        scalar_analysis=True,
        scalar_synthesis=True,
        vector_analysis=True,
        vector_synthesis=True,
    ).to(device="cpu", dtype=dtype)

    floating_buffers = [
        buffer
        for buffer in state.buffers()
        if buffer is not None and buffer.is_floating_point()
    ]
    integer_buffers = [
        buffer
        for buffer in state.buffers()
        if buffer is not None and not buffer.is_floating_point()
    ]
    assert floating_buffers
    assert all(buffer.dtype == dtype for buffer in floating_buffers)
    assert integer_buffers
    assert all(buffer.dtype == torch.long for buffer in integer_buffers)


def test_analyzed_vector_state_keeps_only_vector_synthesis(
    gl_grid: sg.Grid,
) -> None:
    _, eastward, northward = make_fields(gl_grid)

    spectral = sgt.analyze_vector(eastward, northward, grid=gl_grid)

    assert spectral._state._vector_analysis is None
    assert spectral._state._vector_synthesis is not None
    assert spectral._state._scalar_analysis is None
    assert spectral._state._scalar_synthesis is None
    assert spectral.divergence()._state._scalar_synthesis is not None


def test_analyzed_extended_cc_field_releases_forward_projection() -> None:
    grid = sg.clenshaw_curtis_grid(9, 16)
    field = torch.randn(grid.nlat, grid.nlon, dtype=torch.float64)

    spectral = sgt.analyze(field, grid=grid)

    assert spectral.spec == sg.parse_spectral("T7")
    assert spectral._state._scalar_analysis is None
    assert spectral._state._vector_analysis is None
    assert spectral._state._source_latitude_indices is None
    assert spectral._state._scalar_synthesis is not None
    assert not any(
        isinstance(module, torch_backend._ExtendedCCRealSHT)
        for module in spectral._state.modules()
    )


def test_spectral_regrid_constructs_only_target_synthesis(
    gl_grid: sg.Grid,
    cc_grid: sg.Grid,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    field, eastward, northward = make_fields(gl_grid)
    scalar = sgt.analyze(field, grid=gl_grid)
    vector = sgt.analyze_vector(eastward, northward, grid=gl_grid)
    expected_scalar = sgt.regrid(field, cc_grid, source_grid=gl_grid)
    expected_vector = sgt.regrid_vector(
        eastward,
        northward,
        cc_grid,
        source_grid=gl_grid,
    )

    def fail_analysis(*args: object, **kwargs: object) -> torch.nn.Module:
        raise AssertionError("synthesis-only regrid constructed a source analysis")

    monkeypatch.setattr(torch_backend, "_make_analysis_module", fail_analysis)
    scalar_target = scalar.regrid(cc_grid)
    vector_target = vector.regrid(cc_grid)

    torch.testing.assert_close(
        scalar_target,
        expected_scalar,
        rtol=0.0,
        atol=2.0e-12,
    )
    for actual, expected in zip(
        vector_target,
        expected_vector,
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=2.0e-12)


def test_torch_reusable_filters_skip_true_no_ops(gl_grid: sg.Grid) -> None:
    field, eastward, northward = make_fields(gl_grid)
    scalar = sgt.analyze(field, grid=gl_grid)
    vector = sgt.analyze_vector(eastward, northward, grid=gl_grid)

    assert scalar.filter() is scalar
    assert scalar.filter(scalar.spec) is scalar
    assert vector.filter() is vector
    assert vector.filter(vector.spec) is vector


def test_torch_vector_filter_rejects_t0(gl_grid: sg.Grid) -> None:
    _, eastward, northward = make_fields(gl_grid)
    vector = sgt.analyze_vector(eastward, northward, grid=gl_grid)

    with pytest.raises(ValueError, match="lmax >= 1"):
        vector.filter("T0")


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


def test_cc_spectral_fields_use_full_triangular_band_and_match_functions(
    cc_grid: sg.Grid,
) -> None:
    field, eastward, northward = make_fields(cc_grid)

    spectral = sgt.analyze(field, grid=cc_grid)
    assert spectral.spec == sg.parse_spectral("T15")
    torch.testing.assert_close(
        spectral.synthesize(),
        field,
        rtol=0.0,
        atol=2.0e-12,
    )

    spectral_vector = sgt.analyze_vector(eastward, northward, grid=cc_grid)
    assert spectral_vector.spec == sg.parse_spectral("T15")
    torch.testing.assert_close(
        spectral_vector.vorticity().synthesize(),
        sgt.vorticity(eastward, northward, grid=cc_grid),
        rtol=0.0,
        atol=2.0e-12,
    )


def test_full_cc_spectral_field_intersects_smaller_target_at_t9(
    cc_grid: sg.Grid,
) -> None:
    field, _, _ = make_fields(cc_grid)
    spectral = sgt.analyze(field, grid=cc_grid)
    smaller_target = sg.clenshaw_curtis_grid(17, 20)

    assert spectral.spec == sg.parse_spectral("T15")
    actual = spectral.regrid(smaller_target)
    expected = sgt.regrid(field, smaller_target, "T9", source_grid=cc_grid)
    assert float(expected.abs().max()) > 0.0
    torch.testing.assert_close(
        actual,
        expected,
        rtol=0.0,
        atol=2.0e-12,
    )


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
