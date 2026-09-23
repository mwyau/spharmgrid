# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression coverage for JAX transforms on regular GL ``(L, 2L)`` grids."""

from __future__ import annotations

import cftime
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
from jax import Array, config, grad, jit, jvp, vmap

import spharmgrid as sg
import spharmgrid.jax as sgj
from spharmgrid._ducc import (
    _alm_size,
    alm_degrees,
    alm_orders,
    scalar_synthesis,
)
from spharmgrid._transform import TransformSpec
from spharmgrid.jax import _s2fft_gl_compat
from spharmgrid.jax._backend import (
    _make_transform,
    _scalar_analysis,
    _scalar_synthesis,
    _spin_analysis,
    _spin_synthesis,
    _validate_grid,
)
from tests.optional.jax._fields import as_xarray, scalar_values, vector_values

pytestmark = pytest.mark.jax_x64


def _require_x64() -> None:
    if not config.read("jax_enable_x64"):
        pytest.skip("float64 tests require JAX x64 mode")


def test_gl_2l_scalar_analysis_synthesis_and_round_trip_match_ducc() -> None:
    _require_x64()
    grid = sg.gaussian_grid(8, 16, lon0=11.25, latitude_order="descending")
    values = scalar_values(grid).astype(np.float64)
    field = jnp.asarray(values)
    actual = sgj.filter(field, "T5", grid=grid)
    expected = sg.filter(as_xarray(values, grid), "T5").data

    assert actual.shape == (8, 16)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=5.0e-12)
    transform = _make_transform(grid, grid, None)
    coefficients = _scalar_analysis(field, transform)
    reconstructed = _scalar_synthesis(coefficients, transform)
    assert reconstructed.shape == field.shape
    assert bool(jnp.isfinite(reconstructed).all())
    np.testing.assert_allclose(reconstructed, field, rtol=0.0, atol=5.0e-12)


def test_gl_2l_vector_analysis_synthesis_match_ducc() -> None:
    _require_x64()
    grid = sg.gaussian_grid(8, 16, lon0=11.25, latitude_order="ascending")
    eastward_values, northward_values = vector_values(grid)
    eastward = jnp.asarray(eastward_values, dtype=jnp.float64)
    northward = jnp.asarray(northward_values, dtype=jnp.float64)
    actual = sgj.regrid_vector(eastward, northward, grid, "T5", source_grid=grid)
    expected = sg.regrid_vector(
        as_xarray(eastward_values, grid, "u"),
        as_xarray(northward_values, grid, "v"),
        grid,
        "T5",
    )

    np.testing.assert_allclose(actual[0], expected.u, rtol=0.0, atol=8.0e-12)
    np.testing.assert_allclose(actual[1], expected.v, rtol=0.0, atol=8.0e-12)


@pytest.mark.parametrize("spin", [1, -1])
def test_gl_2l_spin_analysis_synthesis(spin: int) -> None:
    _require_x64()
    grid = sg.gaussian_grid(8, 16, latitude_order="descending")
    eastward, northward = vector_values(grid)
    unit = jnp.asarray(1j, dtype=jnp.complex128)
    spin_field = jnp.asarray(-northward, dtype=jnp.complex128) + (
        spin * unit * jnp.asarray(eastward, dtype=jnp.complex128)
    )
    transform = _make_transform(grid, grid, None)

    coefficients = _spin_analysis(spin_field, transform, spin)
    reconstructed = _spin_synthesis(coefficients, transform, spin)

    np.testing.assert_allclose(reconstructed, spin_field, rtol=0.0, atol=5.0e-12)


@pytest.mark.parametrize(
    "order",
    [
        pytest.param(0, id="m0"),
        pytest.param(1, id="m1"),
        pytest.param(6, id="mL-minus-2"),
        pytest.param(7, id="mL-minus-1"),
    ],
)
def test_scalar_spectral_modes_match_ducc(order: int) -> None:
    _require_x64()
    bandlimit = 8
    grid = sg.gaussian_grid(
        bandlimit, 2 * bandlimit, latitude_order="descending", lon0=0.0
    )
    spec = TransformSpec(0, bandlimit - 1, bandlimit - 1)
    degrees = alm_degrees(spec.lmax, spec.mmax)
    orders = alm_orders(spec.lmax, spec.mmax)
    edge = np.flatnonzero((degrees == order) & (orders == order))
    assert edge.size == 1
    packed = np.zeros((1, _alm_size(spec.lmax, spec.mmax)), dtype=np.complex128)
    value = 0.75 if order == 0 else 0.75 - 0.25j
    packed[0, edge[0]] = value
    field = scalar_synthesis(
        packed,
        spec=spec,
        geometry="GL",
        ntheta=bandlimit,
        nphi=2 * bandlimit,
        phi0=0.0,
        nthreads=1,
    )
    expected = np.zeros((bandlimit, 2 * bandlimit - 1), dtype=np.complex128)
    expected[order, bandlimit - 1 + order] = value
    if order:
        expected[order, bandlimit - 1 - order] = (-1) ** order * np.conj(value)

    actual = _scalar_analysis(jnp.asarray(field), _make_transform(grid, grid, None))

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-12)


def test_gl_2l_nyquist_is_projected_out_and_not_synthesized() -> None:
    _require_x64()
    bandlimit = 8
    grid = sg.gaussian_grid(bandlimit, 2 * bandlimit, latitude_order="descending")
    transform = _make_transform(grid, grid, None)
    nyquist = jnp.broadcast_to(
        (-1.0) ** jnp.arange(2 * bandlimit), (bandlimit, 2 * bandlimit)
    )
    projected = _scalar_analysis(nyquist, transform)
    valid_coefficients = _scalar_analysis(
        jnp.asarray(scalar_values(grid), dtype=jnp.float64), transform
    )
    synthesized = _scalar_synthesis(valid_coefficients, transform)
    centered = jnp.fft.fftshift(
        jnp.fft.fft(synthesized, axis=-1, norm="forward"), axes=-1
    )

    np.testing.assert_allclose(projected, 0.0, rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(centered[..., 0], 0.0, rtol=0.0, atol=2.0e-14)


def test_gl_2l_jit_vmap_jvp_and_grad() -> None:
    _require_x64()
    grid = sg.gaussian_grid(8, 16, lon0=11.25, latitude_order="ascending")
    field = jnp.asarray(scalar_values(grid), dtype=jnp.float64)
    batch = jnp.stack((field, 1.5 * field))
    tangent = jnp.cos(jnp.arange(field.size, dtype=jnp.float64)).reshape(field.shape)

    def operation(values: Array) -> Array:
        return sgj.filter(values, "T5", grid=grid)

    compiled = jit(operation)
    compiled_batch = jit(vmap(operation))
    result = compiled(field)
    batched = compiled_batch(batch)
    _, jvp_result = jvp(operation, (field,), (tangent,))
    transform = _make_transform(grid, grid, None)
    coefficients = _scalar_analysis(field, transform)
    inverse_direction = jnp.ones_like(coefficients)

    def inverse(values: Array) -> Array:
        return _scalar_synthesis(values, transform)

    _, inverse_jvp = jvp(inverse, (coefficients,), (inverse_direction,))
    assert inverse_jvp.shape == field.shape
    np.testing.assert_allclose(
        inverse_jvp,
        inverse(inverse_direction),
        rtol=0.0,
        atol=5.0e-12,
    )
    gradient = grad(lambda values: jnp.sum(operation(values) ** 2))(field)

    assert result.shape == field.shape
    assert batched.shape == batch.shape
    assert jvp_result.shape == field.shape
    assert gradient.shape == field.shape
    assert bool(jnp.isfinite(gradient).all())
    np.testing.assert_allclose(result, operation(field), rtol=0.0, atol=5.0e-12)
    np.testing.assert_allclose(
        batched,
        jnp.stack([operation(frame) for frame in batch]),
        rtol=0.0,
        atol=5.0e-12,
    )
    np.testing.assert_allclose(
        jvp_result,
        operation(tangent),
        rtol=0.0,
        atol=5.0e-12,
    )


@pytest.mark.parametrize("nlon", [14, 17])
def test_gl_2l_shapes_accept_native_gl_and_cc_and_reject_nearby_shapes(
    nlon: int,
) -> None:
    bandlimit = 8
    _validate_grid(sg.gaussian_grid(bandlimit, 2 * bandlimit - 1))
    _validate_grid(sg.gaussian_grid(bandlimit, 2 * bandlimit))
    _validate_grid(sg.clenshaw_curtis_grid(bandlimit + 1, 2 * bandlimit))
    with pytest.raises(ValueError, match="only with shape"):
        _validate_grid(sg.gaussian_grid(bandlimit, nlon))


def test_gl_2l_rejects_an_unverified_s2fft_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _s2fft_gl_compat._internal_modules.cache_clear()
    monkeypatch.setattr(_s2fft_gl_compat, "version", lambda _: "1.4.1")
    try:
        with pytest.raises(
            RuntimeError,
            match="other versions are unsupported until verified.*Found s2fft==1.4.1",
        ):
            _s2fft_gl_compat._internal_modules()
    finally:
        _s2fft_gl_compat._internal_modules.cache_clear()


def test_gl_2l_compatibility_symbols_are_available() -> None:
    otf, samples, quadrature = _s2fft_gl_compat._internal_modules()

    assert callable(otf.forward_latitudinal_step_jax)
    assert callable(otf.inverse_latitudinal_step_jax)
    assert callable(samples.thetas)
    assert callable(quadrature.quad_weights_transform)


def test_gl_2l_xarray_accessor_preserves_calendar_and_leading_dimensions() -> None:
    _require_x64()
    grid = sg.gaussian_grid(8, 16, lon0=22.5, latitude_order="ascending")
    base = scalar_values(grid).astype(np.float64)
    values = np.stack(
        [
            np.stack([(1.0 + 0.1 * time + 0.05 * level) * base for level in range(2)])
            for time in range(3)
        ]
    )
    time_values = np.array(
        [cftime.DatetimeNoLeap(2001, 1, day) for day in (1, 2, 3)],
        dtype=object,
    )
    field = xr.DataArray(
        jnp.asarray(values, dtype=jnp.float64),
        dims=("time", "level", "lat", "lon"),
        coords={
            "time": time_values,
            "level": [1000.0, 850.0],
            "lat": grid.latitude,
            "lon": grid.longitude,
        },
        name="temperature",
        attrs={"units": "K"},
    )
    actual = field.sgj.filter("T5")
    expected = sgj.filter(field.data, "T5", grid=grid)

    assert actual.dims == field.dims
    assert actual.name == field.name
    assert actual.attrs == field.attrs
    assert actual.indexes["time"].calendar == "noleap"
    np.testing.assert_array_equal(actual.coords["time"], field.coords["time"])
    np.testing.assert_array_equal(actual.coords["level"], field.coords["level"])
    np.testing.assert_array_equal(actual.coords["lat"], grid.latitude)
    np.testing.assert_array_equal(actual.coords["lon"], grid.longitude)
    np.testing.assert_allclose(actual.data, expected, rtol=0.0, atol=5.0e-12)
