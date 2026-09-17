"""Optional Torch filtering and regridding tests."""

from __future__ import annotations

import numpy as np
import pytest

import spharmgrid as sg
import spharmgrid.torch as sgt
from tests.optional.torch.conftest import (
    as_xarray,
    make_axisymmetric_wind,
    make_fields,
    make_nonaxisymmetric_wind,
    torch,
)


def _tolerance(dtype: torch.dtype) -> tuple[float, float]:
    return (2.0e-11, 2.0e-10) if dtype == torch.float64 else (3.0e-5, 3.0e-4)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("kind", ["gl", "cc"])
def test_filter_matches_cpu_for_explicit_triangular_band(
    dtype: torch.dtype,
    kind: str,
) -> None:
    grid = (
        sg.gaussian_grid(8, 18, latitude_order="descending", lon0=-135.0)
        if kind == "gl"
        else sg.clenshaw_curtis_grid(
            17,
            36,
            latitude_order="descending",
            lon0=-135.0,
        )
    )
    field, _, _ = make_fields(grid, dtype)
    expected = sg.filter(as_xarray(field, grid), "T2").values
    actual = sgt.filter(field, "T2", grid=grid)
    rtol, atol = _tolerance(dtype)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_filter_selection_and_taper_are_not_shape_only(
    dtype: torch.dtype,
    gl_grid: sg.Grid,
) -> None:
    latitude = torch.as_tensor(
        np.sin(np.deg2rad(gl_grid.latitude))[:, None],
        dtype=dtype,
    )
    ones = torch.ones((gl_grid.nlat, gl_grid.nlon), dtype=dtype)
    degree_two = (3.0 * latitude**2 - 1.0) / 2.0 * ones
    low_field = ones + 2.0 * latitude * ones

    low = sgt.filter(low_field + 3.0 * degree_two, "T1", grid=gl_grid)
    band = sgt.filter(low_field + 3.0 * degree_two, "T2-2", grid=gl_grid)
    explicit_band = sgt.filter(
        low_field + 3.0 * degree_two,
        grid=gl_grid,
        lmin=2,
        lmax=2,
    )
    tapered = sgt.filter(degree_two, "T2", grid=gl_grid, taper=0.1)
    rtol, atol = _tolerance(dtype)
    np.testing.assert_allclose(
        low.numpy(),
        low_field.numpy(),
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        band.numpy(),
        (3.0 * degree_two).numpy(),
        rtol=rtol,
        atol=atol,
    )
    torch.testing.assert_close(explicit_band, band, rtol=rtol, atol=atol)
    np.testing.assert_allclose(
        tapered.numpy(),
        (0.1 * degree_two).numpy(),
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("source_kind", ["gl", "cc"])
@pytest.mark.parametrize("target_kind", ["gl", "cc"])
def test_scalar_regrid_matches_cpu(
    dtype: torch.dtype,
    source_kind: str,
    target_kind: str,
) -> None:
    source = (
        sg.gaussian_grid(8, 18, latitude_order="descending", lon0=-135.0)
        if source_kind == "gl"
        else sg.clenshaw_curtis_grid(
            17,
            36,
            latitude_order="descending",
            lon0=-135.0,
        )
    )
    target = (
        sg.gaussian_grid(10, 20, latitude_order="ascending", lon0=75.0)
        if target_kind == "gl"
        else sg.clenshaw_curtis_grid(
            19,
            40,
            latitude_order="ascending",
            lon0=75.0,
        )
    )
    field, _, _ = make_fields(source, dtype)
    expected = sg.regrid(as_xarray(field, source), target, "T2").values
    actual = sgt.regrid(field, target, "T2", source_grid=source)
    rtol, atol = _tolerance(dtype)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_vector_regrid_matches_cpu_and_uses_geographic_vector_semantics(
    dtype: torch.dtype,
    cc_grid: sg.Grid,
) -> None:
    source = cc_grid
    target = sg.gaussian_grid(10, 20, latitude_order="descending", lon0=-75.0)
    u, v = make_axisymmetric_wind(source, dtype)
    expected = sg.regrid_vector(
        as_xarray(u, source, "u"),
        as_xarray(v, source, "v"),
        target,
        "T2",
    )
    actual_u, actual_v = sgt.regrid_vector(
        u,
        v,
        target,
        "T2",
        source_grid=source,
    )
    rtol, atol = _tolerance(dtype)
    np.testing.assert_allclose(
        actual_u.numpy(),
        expected.u.values,
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        actual_v.numpy(),
        expected.v.values,
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_vector_regrid_matches_cpu_for_nonaxisymmetric_longitude_phase(
    dtype: torch.dtype,
) -> None:
    source = sg.gaussian_grid(8, 18, latitude_order="ascending", lon0=45.0)
    target = sg.gaussian_grid(10, 20, latitude_order="descending", lon0=-75.0)
    u, v = make_nonaxisymmetric_wind(source, dtype)
    expected = sg.regrid_vector(
        as_xarray(u, source, "u"),
        as_xarray(v, source, "v"),
        target,
        "T2",
    )
    actual_u, actual_v = sgt.regrid_vector(
        u,
        v,
        target,
        "T2",
        source_grid=source,
    )
    rtol, atol = _tolerance(dtype)
    np.testing.assert_allclose(
        actual_u.numpy(), expected.u.values, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        actual_v.numpy(), expected.v.values, rtol=rtol, atol=atol
    )


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("target_kind", ["gl", "cc"])
def test_vector_regrid_matches_cpu_for_nonaxisymmetric_cc_source(
    dtype: torch.dtype,
    target_kind: str,
) -> None:
    source = sg.clenshaw_curtis_grid(
        17,
        36,
        latitude_order="descending",
        lon0=37.0,
    )
    target = (
        sg.gaussian_grid(
            10,
            20,
            latitude_order="ascending",
            lon0=-83.0,
        )
        if target_kind == "gl"
        else sg.clenshaw_curtis_grid(
            19,
            40,
            latitude_order="ascending",
            lon0=-83.0,
        )
    )
    u, v = make_nonaxisymmetric_wind(source, dtype)
    expected = sg.regrid_vector(
        as_xarray(u, source, "u"),
        as_xarray(v, source, "v"),
        target,
        "T2",
    )
    actual_u, actual_v = sgt.regrid_vector(
        u,
        v,
        target,
        "T2",
        source_grid=source,
    )
    rtol, atol = _tolerance(dtype)
    np.testing.assert_allclose(
        actual_u.numpy(), expected.u.values, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        actual_v.numpy(), expected.v.values, rtol=rtol, atol=atol
    )


def test_leading_dimensions_are_preserved(gl_grid: sg.Grid) -> None:
    field, u, v = make_fields(gl_grid)
    batched = field.unsqueeze(0).expand(2, -1, -1).clone()
    result = sgt.filter(batched, "T2", grid=gl_grid)
    expected = sgt.filter(field, "T2", grid=gl_grid)
    assert result.shape == (2, gl_grid.nlat, gl_grid.nlon)
    torch.testing.assert_close(result[0], expected)
    torch.testing.assert_close(result[1], expected)

    vector_u = u.unsqueeze(0).expand(2, -1, -1).clone()
    vector_v = v.unsqueeze(0).expand(2, -1, -1).clone()
    output_u, output_v = sgt.regrid_vector(
        vector_u,
        vector_v,
        gl_grid,
        "T2",
        source_grid=gl_grid,
    )
    assert output_u.shape == vector_u.shape
    assert output_v.shape == vector_v.shape


def test_unsupported_bandwidths_raise_instead_of_clamping(
    cc_grid: sg.Grid,
    gl_grid: sg.Grid,
) -> None:
    field, _, _ = make_fields(cc_grid)
    with pytest.raises(ValueError, match="full spharmgrid.*filter, regrid"):
        sgt.filter(field, grid=cc_grid)
    with pytest.raises(ValueError, match="verified.*T8"):
        sgt.filter(field, "T9", grid=cc_grid)
    with pytest.raises(ValueError, match="exceeds"):
        sgt.filter(make_fields(gl_grid)[0], "T99", grid=gl_grid)
