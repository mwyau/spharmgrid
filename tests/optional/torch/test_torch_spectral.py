"""Optional Torch filtering and regridding tests."""

from __future__ import annotations

import numpy as np
import pytest

import spharmgrid as sg
import spharmgrid.torch as sgt
from tests.optional.torch.conftest import (
    as_xarray,
    make_fields,
    make_nonaxisymmetric_wind,
    torch,
)


def _spectral_tolerances(
    dtype: torch.dtype,
    *,
    vector: bool = False,
) -> tuple[float, float]:
    """Return measured T2 transform tolerances with platform margin.

    Across GL/CC regrids and filters, the largest measured absolute errors were
    9e-15 for float64 and 8e-7 for float32.  Vector float32 regrids reached
    3.2e-5 relative error at small but meaningful field values.
    """
    if dtype == torch.float64:
        return 1.0e-12, 1.0e-13
    return (3.0e-4, 5.0e-6) if vector else (5.0e-6, 5.0e-6)


def _assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor | np.ndarray,
    dtype: torch.dtype,
    *,
    vector: bool = False,
) -> None:
    rtol, atol = _spectral_tolerances(dtype, vector=vector)
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    else:
        np.testing.assert_allclose(
            actual.detach().cpu().numpy(),
            expected,
            rtol=rtol,
            atol=atol,
        )


def _regrid_grid(kind: str, *, source: bool) -> sg.Grid:
    latitude_order = "ascending" if source else "descending"
    lon0 = 45.0 if source else -75.0
    if kind == "gl":
        nlat, nlon = (8, 18) if source else (10, 20)
        return sg.gaussian_grid(
            nlat,
            nlon,
            latitude_order=latitude_order,
            lon0=lon0,
        )
    nlat, nlon = (17, 36) if source else (19, 40)
    return sg.clenshaw_curtis_grid(
        nlat,
        nlon,
        latitude_order=latitude_order,
        lon0=lon0,
    )


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("kind", ["gl", "cc"])
def test_filter_matches_ducc_xarray_for_explicit_triangular_band(
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
    _assert_close(actual, expected, dtype)


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
    _assert_close(low, low_field, dtype)
    _assert_close(band, 3.0 * degree_two, dtype)
    _assert_close(explicit_band, band, dtype)
    _assert_close(tapered, 0.1 * degree_two, dtype)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("source_kind", ["gl", "cc"])
@pytest.mark.parametrize("target_kind", ["gl", "cc"])
def test_scalar_regrid_matches_analytic_target_field(
    dtype: torch.dtype,
    source_kind: str,
    target_kind: str,
) -> None:
    source = _regrid_grid(source_kind, source=True)
    target = _regrid_grid(target_kind, source=False)
    field, _, _ = make_fields(source, dtype)
    expected, _, _ = make_fields(target, dtype)
    actual = sgt.regrid(field, target, "T2", source_grid=source)
    _assert_close(actual, expected, dtype)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_scalar_regrid_matches_ducc_xarray(
    dtype: torch.dtype,
) -> None:
    source = _regrid_grid("gl", source=True)
    target = _regrid_grid("cc", source=False)
    field, _, _ = make_fields(source, dtype)
    expected = sg.regrid(
        as_xarray(field, source),
        target,
        "T2",
    )
    actual = sgt.regrid(field, target, "T2", source_grid=source)
    _assert_close(actual, expected.values, dtype)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("source_kind", ["gl", "cc"])
@pytest.mark.parametrize("target_kind", ["gl", "cc"])
def test_vector_regrid_matches_analytic_target_field(
    dtype: torch.dtype,
    source_kind: str,
    target_kind: str,
) -> None:
    source = _regrid_grid(source_kind, source=True)
    target = _regrid_grid(target_kind, source=False)
    u, v = make_nonaxisymmetric_wind(source, dtype)
    expected_u, expected_v = make_nonaxisymmetric_wind(target, dtype)
    actual_u, actual_v = sgt.regrid_vector(
        u,
        v,
        target,
        "T2",
        source_grid=source,
    )
    _assert_close(actual_u, expected_u, dtype, vector=True)
    _assert_close(actual_v, expected_v, dtype, vector=True)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_vector_regrid_matches_ducc_xarray(dtype: torch.dtype) -> None:
    source = _regrid_grid("cc", source=True)
    target = _regrid_grid("gl", source=False)
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
    _assert_close(actual_u, expected.u.values, dtype, vector=True)
    _assert_close(actual_v, expected.v.values, dtype, vector=True)


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
    with pytest.raises(ValueError, match="Full-domain operations.*filter, regrid"):
        sgt.filter(field, grid=cc_grid)
    with pytest.raises(ValueError, match="CC triangular bands through T8"):
        sgt.filter(field, "T9", grid=cc_grid)
    with pytest.raises(ValueError, match="exceeds"):
        sgt.filter(make_fields(gl_grid)[0], "T99", grid=gl_grid)
