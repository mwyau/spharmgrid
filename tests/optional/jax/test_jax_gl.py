# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression coverage for arbitrary regular GL longitude counts in JAX."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array, config, grad, jit, jvp, vmap

import spharmgrid as sg
import spharmgrid.jax as sgj
from spharmgrid._ducc import (
    _alm_size,
    alm_degrees,
    alm_orders,
    geometry_for,
)
from spharmgrid._ducc import (
    scalar_analysis as ducc_scalar_analysis,
)
from spharmgrid._ducc import (
    scalar_synthesis as ducc_scalar_synthesis,
)
from spharmgrid._transform import TransformSpec
from spharmgrid.grids import grid_layout
from spharmgrid.jax import _s2fft_gl_compat
from spharmgrid.jax._backend import (
    _make_transform,
    _scalar_analysis,
    _scalar_synthesis,
    _spin_analysis,
    _spin_synthesis,
    _validate_grid,
    _vector_analysis,
    _vector_synthesis,
)
from spharmgrid.spectral import parse_spectral, resolve_transform_spec
from tests.optional.jax._fields import as_xarray, scalar_values, vector_values

pytestmark = pytest.mark.jax_x64


def _require_x64() -> None:
    if not config.read("jax_enable_x64"):
        pytest.skip("float64 tests require JAX x64 mode")


def _ducc_analysis(field: np.ndarray, grid: sg.Grid, spec: TransformSpec) -> np.ndarray:
    layout = grid_layout(grid)
    canonical = np.take(field, layout.latitude.canonical_indices, axis=-2)
    canonical = np.take(canonical, layout.longitude.canonical_indices, axis=-1)
    return ducc_scalar_analysis(
        canonical,
        spec=spec,
        geometry=geometry_for(grid),
        phi0=layout.phi0_radians,
        nthreads=1,
    )[0]


def _ducc_synthesis(
    coefficients: np.ndarray, grid: sg.Grid, spec: TransformSpec
) -> np.ndarray:
    layout = grid_layout(grid)
    canonical = ducc_scalar_synthesis(
        coefficients[None, :],
        spec=spec,
        geometry=geometry_for(grid),
        ntheta=grid.nlat,
        nphi=grid.nlon,
        phi0=layout.phi0_radians,
        nthreads=1,
    )
    values = np.take(canonical, layout.latitude.restore_indices, axis=-2)
    return np.take(values, layout.longitude.restore_indices, axis=-1)


def _centered_coefficients(
    packed: np.ndarray, spec: TransformSpec, bandlimit: int
) -> np.ndarray:
    centered = np.zeros((bandlimit, 2 * bandlimit - 1), dtype=np.complex128)
    degrees = alm_degrees(spec.lmax, spec.mmax)
    orders = alm_orders(spec.lmax, spec.mmax)
    center = bandlimit - 1
    for value, degree, order in zip(packed, degrees, orders, strict=True):
        degree_index = int(degree)
        order_index = int(order)
        coefficient = complex(value)
        centered[degree_index, center + order_index] = coefficient
        if order_index:
            centered[degree_index, center - order_index] = (
                -1
            ) ** order_index * np.conj(coefficient)
    return centered


@pytest.mark.parametrize("nlon", [2, 9, 12, 15, 16, 18, 20])
def test_scalar_analysis_synthesis_and_round_trip_match_ducc(nlon: int) -> None:
    _require_x64()
    bandlimit = 8
    grid = sg.gaussian_grid(bandlimit, nlon, lon0=0.0, latitude_order="descending")
    spec = resolve_transform_spec(grid, grid, None)
    transform = _make_transform(grid, grid, None)
    values = np.random.default_rng(nlon).normal(size=(bandlimit, nlon))
    expected_packed = _ducc_analysis(values, grid, spec)
    expected_coefficients = _centered_coefficients(expected_packed, spec, bandlimit)
    actual_coefficients = _scalar_analysis(jnp.asarray(values), transform)

    assert actual_coefficients.shape == (bandlimit, 2 * bandlimit - 1)
    np.testing.assert_allclose(
        actual_coefficients, expected_coefficients, rtol=0.0, atol=2.0e-12
    )
    actual_values = _scalar_synthesis(jnp.asarray(expected_coefficients), transform)
    expected_values = _ducc_synthesis(expected_packed, grid, spec)
    np.testing.assert_allclose(actual_values, expected_values, rtol=0.0, atol=2.0e-12)

    analyzed = sgj.analyze(jnp.asarray(values), grid=grid)
    assert analyzed.spec == spec
    expected_mmax = min(bandlimit - 1, (nlon - 1) // 2)
    assert analyzed.spec.mmax == expected_mmax
    orders = np.abs(np.arange(-(bandlimit - 1), bandlimit))
    np.testing.assert_array_equal(
        np.asarray(analyzed._coefficients)[:, orders > analyzed.spec.mmax], 0.0
    )
    np.testing.assert_allclose(analyzed.synthesize(), expected_values, atol=2.0e-12)



@pytest.mark.parametrize("notation", ["T4", "T6x4", "T4x2", "R2"])
def test_analyzed_coefficients_follow_the_resolved_selection(notation: str) -> None:
    _require_x64()
    grid = sg.gaussian_grid(8, 9, latitude_order="descending")
    values = scalar_values(grid).astype(np.float64)
    reference = as_xarray(values, grid)
    field = jnp.asarray(values)
    requested = parse_spectral(notation)
    expected_spec = resolve_transform_spec(grid, grid, requested)
    analyzed = sgj.analyze(field, notation, grid=grid)

    assert analyzed.spec == expected_spec
    degrees = np.arange(8)[:, None]
    orders = np.abs(np.arange(-7, 8))[None, :]
    inside = (
        (degrees >= expected_spec.lmin)
        & (degrees <= expected_spec.lmax)
        & (orders <= expected_spec.mmax)
    )
    if expected_spec.truncation == "rhomboidal":
        inside = inside & (degrees - orders <= expected_spec.lmax - expected_spec.mmax)
    np.testing.assert_array_equal(np.asarray(analyzed._coefficients)[~inside], 0.0)
    expected = sg.filter(reference, notation)
    np.testing.assert_allclose(analyzed.synthesize(), expected, rtol=0.0, atol=2e-12)


def test_even_longitude_nyquist_is_excluded_in_analysis_and_synthesis() -> None:
    _require_x64()
    nlon = 12
    bandlimit = 8
    grid = sg.gaussian_grid(bandlimit, nlon, latitude_order="descending")
    transform = _make_transform(grid, grid, None)
    nyquist = jnp.broadcast_to((-1.0) ** jnp.arange(nlon), (bandlimit, nlon))
    projected = _scalar_analysis(nyquist, transform)
    field = jnp.asarray(scalar_values(grid), dtype=jnp.float64)
    valid = _scalar_analysis(field, transform)
    synthesized = _scalar_synthesis(valid, transform)
    nyquist_bin = jnp.fft.rfft(synthesized, axis=-1, norm="forward")[..., nlon // 2]

    np.testing.assert_allclose(projected, 0.0, rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(nyquist_bin, 0.0, rtol=0.0, atol=2.0e-14)


@pytest.mark.parametrize("nlon", [9, 20])
def test_first_unavailable_order_is_zero_in_s2fft_domain(nlon: int) -> None:
    _require_x64()
    bandlimit = 8
    mmax = min(bandlimit - 1, (nlon - 1) // 2)
    next_order = mmax + 1
    grid = sg.gaussian_grid(bandlimit, nlon, latitude_order="descending")
    transform = _make_transform(grid, grid, None)

    if nlon == 9:
        # The sampled +5 mode aliases to -4 on an odd 9-point grid. Its +5
        # coefficient remains absent from the available spherical domain.
        spec = TransformSpec(0, bandlimit - 1, next_order, "trapezoidal")
        packed = np.zeros((1, _alm_size(spec.lmax, spec.mmax)), dtype=np.complex128)
        index = np.flatnonzero(
            (alm_degrees(spec.lmax, spec.mmax) == next_order)
            & (alm_orders(spec.lmax, spec.mmax) == next_order)
        )
        assert index.size == 1
        packed[0, index[0]] = 0.5 + 0.25j
        field = ducc_scalar_synthesis(
            packed,
            spec=spec,
            geometry="GL",
            ntheta=bandlimit,
            nphi=nlon,
            phi0=0.0,
            nthreads=1,
        )
        coefficients = _scalar_analysis(jnp.asarray(field), transform)
        np.testing.assert_allclose(
            coefficients[:, bandlimit - 1 + next_order],
            0.0,
            rtol=0.0,
            atol=2.0e-14,
        )
        assert float(jnp.max(jnp.abs(coefficients[:, bandlimit - 1 - mmax]))) > 0.0
    else:
        # These next modes are unique physical FFT bins but lie beyond the
        # fixed spherical domain selected by latitude bandlimit.
        longitude = 2.0 * np.pi * np.arange(nlon) / nlon
        field = np.cos(next_order * longitude)[None, :] * np.ones((bandlimit, 1))
        coefficients = _scalar_analysis(jnp.asarray(field), transform)
        np.testing.assert_allclose(coefficients, 0.0, rtol=0.0, atol=2.0e-14)


@pytest.mark.parametrize("spin", [1, -1])
@pytest.mark.parametrize("nlon", [9, 16, 20])
def test_spin_and_vector_analysis_synthesis(spin: int, nlon: int) -> None:
    _require_x64()
    grid = sg.gaussian_grid(8, nlon, latitude_order="ascending", lon0=37.0)
    eastward, northward = vector_values(grid)
    unit = jnp.asarray(1j, dtype=jnp.complex128)
    spin_field = -jnp.asarray(northward, dtype=jnp.float64) + (
        spin * unit * jnp.asarray(eastward, dtype=jnp.float64)
    )
    transform = _make_transform(grid, grid, None)
    spin_coefficients = _spin_analysis(spin_field, transform, spin)
    reconstructed_spin = _spin_synthesis(spin_coefficients, transform, spin)
    np.testing.assert_allclose(reconstructed_spin, spin_field, rtol=0.0, atol=5.0e-12)

    coefficients = _vector_analysis(
        jnp.asarray(eastward, dtype=jnp.float64),
        jnp.asarray(northward, dtype=jnp.float64),
        transform,
    )
    actual = _vector_synthesis(coefficients, transform)
    np.testing.assert_allclose(actual[0], eastward, rtol=0.0, atol=5.0e-12)
    np.testing.assert_allclose(actual[1], northward, rtol=0.0, atol=5.0e-12)

    spectral_vector = sgj.analyze_vector(
        jnp.asarray(eastward, dtype=jnp.float64),
        jnp.asarray(northward, dtype=jnp.float64),
        grid=grid,
    )
    assert spectral_vector.spec == transform.spec
    orders = np.abs(np.arange(-7, 8))
    np.testing.assert_array_equal(
        np.asarray(spectral_vector._coefficients)[..., orders > transform.spec.mmax],
        0.0,
    )


def test_regular_gl_grid_validation_has_no_longitude_upper_bound() -> None:
    _validate_grid(sg.gaussian_grid(8, 32))


def test_cc_mwss_shape_validation_is_unchanged() -> None:
    with pytest.raises(ValueError, match="CC/MWSS grids only with shape"):
        _validate_grid(sg.clenshaw_curtis_grid(9, 15))


@pytest.mark.parametrize("nlon", [9, 12])
def test_regular_gl_jit_vmap_jvp_and_grad(nlon: int) -> None:
    _require_x64()
    grid = sg.gaussian_grid(8, nlon, lon0=11.25, latitude_order="ascending")
    field = jnp.asarray(scalar_values(grid), dtype=jnp.float64)
    batch = jnp.stack((field, 1.5 * field))
    tangent = jnp.cos(jnp.arange(field.size, dtype=jnp.float64)).reshape(field.shape)

    def operation(values: Array) -> Array:
        return sgj.filter(values, grid=grid)

    result = jit(operation)(field)
    batched = jit(vmap(operation))(batch)
    _, jvp_result = jvp(operation, (field,), (tangent,))
    cotangent = jnp.sin(jnp.arange(field.size, dtype=jnp.float64)).reshape(field.shape)
    gradient = grad(lambda values: jnp.vdot(operation(values), cotangent))(field)
    directional = jnp.vdot(cotangent, jvp_result)

    transform = _make_transform(grid, grid, None)
    coefficients = _scalar_analysis(field, transform)
    inverse_direction = jnp.ones_like(coefficients)

    def inverse(values: Array) -> Array:
        return _scalar_synthesis(values, transform)

    _, inverse_jvp = jvp(inverse, (coefficients,), (inverse_direction,))

    assert result.shape == field.shape
    assert batched.shape == batch.shape
    assert jvp_result.shape == field.shape
    assert gradient.shape == field.shape
    assert inverse_jvp.shape == field.shape
    np.testing.assert_allclose(result, operation(field), rtol=0.0, atol=5.0e-12)
    np.testing.assert_allclose(
        batched,
        jnp.stack([operation(frame) for frame in batch]),
        rtol=0.0,
        atol=5.0e-12,
    )
    np.testing.assert_allclose(jvp_result, operation(tangent), atol=5.0e-12)
    np.testing.assert_allclose(
        jnp.vdot(gradient, tangent), directional, rtol=0.0, atol=1.0e-11
    )
    np.testing.assert_allclose(
        inverse_jvp, inverse(inverse_direction), rtol=0.0, atol=5.0e-12
    )


def test_regular_gl_compatibility_rejects_an_unverified_s2fft_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _s2fft_gl_compat._internal_modules.cache_clear()
    monkeypatch.setattr(_s2fft_gl_compat, "version", lambda _: "1.4.1")
    try:
        with pytest.raises(
            RuntimeError,
            match="another version is unsupported until verified.*Found s2fft==1.4.1",
        ):
            _s2fft_gl_compat._internal_modules()
    finally:
        _s2fft_gl_compat._internal_modules.cache_clear()


def test_regular_gl_xarray_accessor_matches_direct_api() -> None:
    _require_x64()
    grid = sg.gaussian_grid(8, 20, lon0=22.5, latitude_order="ascending")
    values = scalar_values(grid).astype(np.float64)
    field = sgj.device_put(as_xarray(values, grid))

    actual = field.sgj.filter()
    expected = sgj.filter(field.data, grid=grid)

    assert actual.dims == field.dims
    np.testing.assert_array_equal(actual.coords["lat"], field.coords["lat"])
    np.testing.assert_array_equal(actual.coords["lon"], field.coords["lon"])
    np.testing.assert_allclose(actual.data, expected, rtol=0.0, atol=5.0e-12)
