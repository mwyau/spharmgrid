# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the correctness and JAX-transformation matrix for GL ``(L, 2L)``."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import cast

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp
import numpy as np
import s2fft
import xarray as xr
from _experiment import METHODS, select_method
from jax import Array, config, default_backend, devices, grad, jit, jvp, vmap

import spharmgrid as sg
import spharmgrid.jax as sgj
from spharmgrid._ducc import (
    _alm_size,
    alm_degrees,
    alm_orders,
    scalar_analysis,
    scalar_synthesis,
    vector_analysis,
    vector_synthesis,
)
from spharmgrid._transform import TransformSpec
from spharmgrid.grids import Grid, GridLayout, grid_layout
from spharmgrid.jax._backend import (
    _JaxTransform,
    _make_transform,
    _precomputes,
    _scalar_analysis,
    _scalar_synthesis,
    _spin_analysis,
    _spin_synthesis,
    _vector_analysis,
    _vector_synthesis,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = (
    "low_degree_dominated",
    "high_degree_dominated",
    "full_spectrum_random_phase",
    "full_spectrum_random_amplitude",
    "large_dynamic_range",
)
DEFAULT_SIZES = (4, 8, 16, 32, 64, 128)


def _metrics(actual: object, expected: object) -> dict[str, float | int | None]:
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    difference = np.abs(actual_array - expected_array)
    reference_abs = np.abs(expected_array)
    max_absolute = float(np.max(difference, initial=0.0))
    rms_absolute = float(np.sqrt(np.mean(difference**2))) if difference.size else 0.0
    reference_rms = (
        float(np.sqrt(np.mean(reference_abs**2))) if reference_abs.size else 0.0
    )
    relative_rms = rms_absolute / reference_rms if reference_rms > 0.0 else None
    threshold = max(float(np.max(reference_abs, initial=0.0)) * 1.0e-8, 1.0e-300)
    meaningful = reference_abs > threshold
    max_relative = (
        float(np.max(difference[meaningful] / reference_abs[meaningful]))
        if np.any(meaningful)
        else None
    )
    return {
        "max_absolute": max_absolute,
        "rms_absolute": rms_absolute,
        "relative_rms": relative_rms,
        "max_relative": max_relative,
        "reference_rms": reference_rms,
        "relative_threshold": threshold,
        "n_values": int(difference.size),
    }


def _row(
    method: str,
    bandlimit: int,
    spin: str,
    fixture: str,
    operation: str,
    actual: object,
    expected: object,
) -> dict[str, object]:
    values = np.asarray(actual)
    return {
        "candidate": method,
        "L": bandlimit,
        "spin": spin,
        "fixture": fixture,
        "operation": operation,
        "dtype": str(values.dtype),
        "device": default_backend(),
        **_metrics(actual, expected),
    }


def _metadata() -> dict[str, object]:
    distribution = importlib.metadata.distribution("s2fft")
    direct_url_text = distribution.read_text("direct_url.json")
    direct_url = json.loads(direct_url_text) if direct_url_text else None
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "spharmgrid_commit": commit,
        "s2fft_version": distribution.version,
        "s2fft_direct_url": direct_url,
        "jax_version": importlib.metadata.version("jax"),
        "jaxlib_version": importlib.metadata.version("jaxlib"),
        "jax_x64_enabled": bool(config.read("jax_enable_x64")),
        "jax_backend": default_backend(),
        "jax_devices": [
            {
                "platform": device.platform,
                "device_kind": getattr(device, "device_kind", None),
                "id": device.id,
            }
            for device in devices()
        ],
        "ducc0_version": importlib.metadata.version("ducc0"),
    }


def _spec(bandlimit: int) -> TransformSpec:
    return TransformSpec(0, bandlimit - 1, bandlimit - 1)


def _dense_coefficients(packed: np.ndarray, bandlimit: int) -> np.ndarray:
    """Convert DUCC's packed real-field coefficients to S2FFT's signed-m array."""
    packed = np.asarray(packed).reshape(-1)
    dense = np.zeros((bandlimit, 2 * bandlimit - 1), dtype=np.complex128)
    degrees = alm_degrees(bandlimit - 1, bandlimit - 1)
    orders = alm_orders(bandlimit - 1, bandlimit - 1)
    for index, (degree, order) in enumerate(zip(degrees, orders, strict=True)):
        degree_index = int(degree)
        order_index = int(order)
        dense[degree_index, bandlimit - 1 + order_index] = complex(packed[index])
        if order:
            dense[degree_index, bandlimit - 1 - order_index] = complex(
                (-1) ** order_index * np.conj(packed[index])
            )
    return dense


def _canonicalize(values: np.ndarray, layout: GridLayout) -> np.ndarray:
    result = np.take(values, layout.latitude.canonical_indices, axis=-2)
    return np.take(result, layout.longitude.canonical_indices, axis=-1)


def _restore(values: np.ndarray, layout: GridLayout) -> np.ndarray:
    result = np.take(values, layout.latitude.restore_indices, axis=-2)
    return np.take(result, layout.longitude.restore_indices, axis=-1)


def _scalar_field_from_coefficients(
    coefficients: np.ndarray, bandlimit: int, grid: Grid
) -> np.ndarray:
    layout = grid_layout(grid)
    canonical = scalar_synthesis(
        coefficients,
        spec=_spec(bandlimit),
        geometry="GL",
        ntheta=bandlimit,
        nphi=2 * bandlimit,
        phi0=layout.phi0_radians,
        nthreads=1,
    )
    return _restore(canonical, layout)


def _vector_fields_from_coefficients(
    electric: np.ndarray, magnetic: np.ndarray, bandlimit: int, grid: Grid
) -> tuple[np.ndarray, np.ndarray]:
    layout = grid_layout(grid)
    canonical_u, canonical_v = vector_synthesis(
        electric,
        magnetic,
        spec=_spec(bandlimit),
        geometry="GL",
        ntheta=bandlimit,
        nphi=2 * bandlimit,
        phi0=layout.phi0_radians,
        nthreads=1,
    )
    return _restore(canonical_u, layout), _restore(canonical_v, layout)


def _random_spectrum(
    bandlimit: int, fixture: str, seed: int, *, vector: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    degrees = alm_degrees(bandlimit - 1, bandlimit - 1)
    orders = alm_orders(bandlimit - 1, bandlimit - 1)
    if fixture == "low_degree_dominated":
        active = degrees <= min(2, bandlimit - 1)
    elif fixture == "high_degree_dominated":
        active = degrees >= max(1, bandlimit - 2)
    else:
        active = np.ones(degrees.shape, dtype=bool)

    def draw() -> np.ndarray:
        values = (
            rng.standard_normal(degrees.size) + 1j * rng.standard_normal(degrees.size)
        ) / np.sqrt(2.0)
        values[orders == 0] = rng.standard_normal(np.count_nonzero(orders == 0))
        if fixture == "full_spectrum_random_amplitude":
            values = np.exp(1j * np.angle(values)) * rng.uniform(
                0.1, 1.0, size=values.size
            )
        elif fixture == "large_dynamic_range":
            values *= 10.0 ** rng.uniform(-8.0, 8.0, size=values.size)
        return np.where(active, values, 0.0).astype(np.complex128)

    if not vector:
        coefficients = draw()
        coefficients[degrees == 0] = coefficients[degrees == 0].real
        return coefficients[np.newaxis, :]

    electric = draw()
    magnetic = draw()
    if fixture == "large_dynamic_range":
        # Keep the vector fixture's peak magnitudes comparable to the scalar one.
        electric = np.clip(electric, -1.0e8, 1.0e8)
        magnetic = np.clip(magnetic, -1.0e8, 1.0e8)
    electric[orders == 0] = electric[orders == 0].real
    magnetic[orders == 0] = magnetic[orders == 0].real
    electric[degrees == 0] = 0.0
    magnetic[degrees == 0] = 0.0
    return electric, magnetic


def _analyze_scalar_reference(
    values: np.ndarray, bandlimit: int, grid: Grid
) -> np.ndarray:
    layout = grid_layout(grid)
    canonical = _canonicalize(values, layout)
    packed = scalar_analysis(
        canonical,
        spec=_spec(bandlimit),
        geometry="GL",
        phi0=layout.phi0_radians,
        nthreads=1,
    )[0]
    return _dense_coefficients(packed, bandlimit)


def _analyze_vector_reference(
    u: np.ndarray, v: np.ndarray, bandlimit: int, grid: Grid
) -> np.ndarray:
    layout = grid_layout(grid)
    canonical_u = _canonicalize(u, layout)
    canonical_v = _canonicalize(v, layout)
    packed = vector_analysis(
        canonical_u,
        canonical_v,
        spec=_spec(bandlimit),
        geometry="GL",
        phi0=layout.phi0_radians,
        nthreads=1,
    )
    return np.stack(
        (
            _dense_coefficients(packed[0], bandlimit),
            _dense_coefficients(packed[1], bandlimit),
        )
    )


def _analytic_scalar_cases(bandlimit: int) -> list[tuple[int, int]]:
    degrees = sorted(
        {
            0,
            1,
            bandlimit // 2,
            bandlimit - 2,
            bandlimit - 1,
        }
    )
    cases: set[tuple[int, int]] = set()
    for degree in degrees:
        if not 0 <= degree < bandlimit:
            continue
        orders = (0, 1, degree // 2, degree, bandlimit - 2, bandlimit - 1)
        cases.update((degree, order) for order in orders if 0 <= order <= degree)
    return sorted(cases)


def _make_scalar_mode(bandlimit: int, degree: int, order: int) -> np.ndarray:
    coefficients = np.zeros((1, _alm_size(bandlimit - 1, bandlimit - 1)), np.complex128)
    degrees = alm_degrees(bandlimit - 1, bandlimit - 1)
    orders = alm_orders(bandlimit - 1, bandlimit - 1)
    index = np.flatnonzero((degrees == degree) & (orders == order))[0]
    coefficients[0, index] = 1.0 if order == 0 else 0.7 - 0.3j
    return coefficients


def _analytic_vector_cases(bandlimit: int) -> list[tuple[int, int]]:
    degrees = sorted({1, max(1, bandlimit // 2), max(1, bandlimit - 2), bandlimit - 1})
    cases: set[tuple[int, int]] = set()
    for degree in degrees:
        if not 1 <= degree < bandlimit:
            continue
        orders = (0, 1, degree // 2, degree, bandlimit - 2, bandlimit - 1)
        cases.update((degree, order) for order in orders if 0 <= order <= degree)
    return sorted(cases)


def _make_vector_mode(
    bandlimit: int, degree: int, order: int, *, magnetic_mode: bool
) -> tuple[np.ndarray, np.ndarray]:
    size = _alm_size(bandlimit - 1, bandlimit - 1)
    electric = np.zeros(size, dtype=np.complex128)
    magnetic = np.zeros(size, dtype=np.complex128)
    degrees = alm_degrees(bandlimit - 1, bandlimit - 1)
    orders = alm_orders(bandlimit - 1, bandlimit - 1)
    index = np.flatnonzero((degrees == degree) & (orders == order))[0]
    value = 1.0 if order == 0 else 0.7 - 0.3j
    (magnetic if magnetic_mode else electric)[index] = value
    return electric, magnetic


def _gl_grid_variants(bandlimit: int) -> dict[str, Grid]:
    half_interval = 180.0 / (2 * bandlimit)
    zero = sg.gaussian_grid(
        bandlimit, 2 * bandlimit, lon0=0.0, latitude_order="descending"
    )
    shifted = sg.gaussian_grid(
        bandlimit,
        2 * bandlimit,
        lon0=half_interval,
        latitude_order="ascending",
    )
    wrapped_source = sg.gaussian_grid(
        bandlimit, 2 * bandlimit, lon0=270.0, latitude_order="ascending"
    )
    wrapped = Grid(
        "gl",
        wrapped_source.latitude,
        (wrapped_source.longitude + 180.0) % 360.0 - 180.0,
    )
    return {"zero_descending": zero, "half_step_ascending": shifted, "wrapped": wrapped}


def _scalar_matrix_rows(sizes: tuple[int, ...]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for bandlimit in sizes:
        grid = _gl_grid_variants(bandlimit)["zero_descending"]
        transform = _make_transform(grid, grid, None)
        for fixture_index, fixture in enumerate(FIXTURES):
            packed = _random_spectrum(
                bandlimit, fixture, 1009 + 31 * bandlimit + fixture_index
            )
            assert isinstance(packed, np.ndarray)
            values = _scalar_field_from_coefficients(packed, bandlimit, grid)
            expected_coefficients = _analyze_scalar_reference(values, bandlimit, grid)
            for method in METHODS:
                select_method(method)
                coefficients = _scalar_analysis(jnp.asarray(values), transform)
                reconstructed = _scalar_synthesis(coefficients, transform)
                rows.append(
                    _row(
                        method,
                        bandlimit,
                        "0",
                        fixture,
                        "scalar_forward_coefficients_vs_ducc",
                        coefficients,
                        expected_coefficients,
                    )
                )
                rows.append(
                    _row(
                        method,
                        bandlimit,
                        "0",
                        fixture,
                        "scalar_inverse_field_vs_ducc_synthesis",
                        reconstructed,
                        values,
                    )
                )

        for degree, order in _analytic_scalar_cases(bandlimit):
            packed = _make_scalar_mode(bandlimit, degree, order)
            values = _scalar_field_from_coefficients(packed, bandlimit, grid)
            expected_coefficients = _analyze_scalar_reference(values, bandlimit, grid)
            fixture = f"analytic_l{degree}_m{order}"
            for method in METHODS:
                select_method(method)
                coefficients = _scalar_analysis(jnp.asarray(values), transform)
                reconstructed = _scalar_synthesis(coefficients, transform)
                rows.append(
                    _row(
                        method,
                        bandlimit,
                        "0",
                        fixture,
                        "scalar_forward_coefficients_vs_ducc",
                        coefficients,
                        expected_coefficients,
                    )
                )
                rows.append(
                    _row(
                        method,
                        bandlimit,
                        "0",
                        fixture,
                        "scalar_inverse_field_vs_ducc_synthesis",
                        reconstructed,
                        values,
                    )
                )

    # Exercise physical longitude origins, wrapped coordinates, and both
    # latitude orders while comparing globally phased coefficients to DUCC.
    bandlimit = 8
    for fixture_index, (fixture, grid) in enumerate(
        _gl_grid_variants(bandlimit).items()
    ):
        packed = _random_spectrum(bandlimit, "full_spectrum_random_phase", 7901)
        assert isinstance(packed, np.ndarray)
        values = _scalar_field_from_coefficients(packed, bandlimit, grid)
        expected_coefficients = _analyze_scalar_reference(values, bandlimit, grid)
        transform = _make_transform(grid, grid, None)
        for method in METHODS:
            select_method(method)
            coefficients = _scalar_analysis(jnp.asarray(values), transform)
            reconstructed = _scalar_synthesis(coefficients, transform)
            rows.append(
                _row(
                    method,
                    bandlimit,
                    "0",
                    fixture,
                    f"longitude_layout_{fixture_index}_coefficients_vs_ducc",
                    coefficients,
                    expected_coefficients,
                )
            )
            rows.append(
                _row(
                    method,
                    bandlimit,
                    "0",
                    fixture,
                    f"longitude_layout_{fixture_index}_field_round_trip",
                    reconstructed,
                    values,
                )
            )
    return rows


def _vector_matrix_rows(sizes: tuple[int, ...]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for bandlimit in sizes:
        grid = _gl_grid_variants(bandlimit)["zero_descending"]
        transform = _make_transform(grid, grid, None)
        cases: list[tuple[str, np.ndarray, np.ndarray]] = []
        for fixture_index, fixture in enumerate(FIXTURES):
            coefficients = _random_spectrum(
                bandlimit,
                fixture,
                2801 + 37 * bandlimit + fixture_index,
                vector=True,
            )
            assert isinstance(coefficients, tuple)
            electric, magnetic = coefficients
            u, v = _vector_fields_from_coefficients(electric, magnetic, bandlimit, grid)
            cases.append((fixture, u, v))
        analytic_cases = _analytic_vector_cases(bandlimit)
        if bandlimit > 32:
            analytic_cases = sorted(
                {
                    (1, 0),
                    (bandlimit // 2, 1),
                    (bandlimit - 2, bandlimit - 2),
                    (bandlimit - 1, bandlimit - 1),
                }
            )
        for degree, order in analytic_cases:
            for magnetic_mode in (False, True):
                electric, magnetic = _make_vector_mode(
                    bandlimit, degree, order, magnetic_mode=magnetic_mode
                )
                u, v = _vector_fields_from_coefficients(
                    electric, magnetic, bandlimit, grid
                )
                mode = "B" if magnetic_mode else "E"
                cases.append((f"analytic_{mode}_l{degree}_m{order}", u, v))

        for fixture, u, v in cases:
            expected_coefficients = _analyze_vector_reference(u, v, bandlimit, grid)
            q_plus = -v + 1j * u
            q_minus = -v - 1j * u
            expected_plus = -expected_coefficients[0] - 1j * expected_coefficients[1]
            expected_minus = expected_coefficients[0] - 1j * expected_coefficients[1]
            for method in METHODS:
                select_method(method)
                coefficients = _vector_analysis(
                    jnp.asarray(u), jnp.asarray(v), transform
                )
                reconstructed_u, reconstructed_v = _vector_synthesis(
                    coefficients, transform
                )
                spin_plus = _spin_analysis(jnp.asarray(q_plus), transform, 1)
                spin_minus = _spin_analysis(jnp.asarray(q_minus), transform, -1)
                spin_plus_inverse = _spin_synthesis(
                    jnp.asarray(expected_plus), transform, 1
                )
                spin_minus_inverse = _spin_synthesis(
                    jnp.asarray(expected_minus), transform, -1
                )
                rows.extend(
                    (
                        _row(
                            method,
                            bandlimit,
                            "±1",
                            fixture,
                            "vector_forward_coefficients_vs_ducc",
                            coefficients,
                            expected_coefficients,
                        ),
                        _row(
                            method,
                            bandlimit,
                            "+1",
                            fixture,
                            "spin_forward_vs_ducc_vector_spectrum",
                            spin_plus,
                            expected_plus,
                        ),
                        _row(
                            method,
                            bandlimit,
                            "-1",
                            fixture,
                            "spin_forward_vs_ducc_vector_spectrum",
                            spin_minus,
                            expected_minus,
                        ),
                        _row(
                            method,
                            bandlimit,
                            "±1",
                            fixture,
                            "vector_inverse_u_vs_ducc_synthesis",
                            reconstructed_u,
                            u,
                        ),
                        _row(
                            method,
                            bandlimit,
                            "±1",
                            fixture,
                            "vector_inverse_v_vs_ducc_synthesis",
                            reconstructed_v,
                            v,
                        ),
                        _row(
                            method,
                            bandlimit,
                            "+1",
                            fixture,
                            "spin_inverse_vs_ducc_vector_synthesis",
                            spin_plus_inverse,
                            q_plus,
                        ),
                        _row(
                            method,
                            bandlimit,
                            "-1",
                            fixture,
                            "spin_inverse_vs_ducc_vector_synthesis",
                            spin_minus_inverse,
                            q_minus,
                        ),
                    )
                )
    return rows


def _analytic_spin_rows(sizes: tuple[int, ...]) -> list[dict[str, object]]:
    """Check signed isolated spin harmonics against native public S2FFT."""
    rows: list[dict[str, object]] = []
    for bandlimit in sizes:
        grid = _gl_grid_variants(bandlimit)["zero_descending"]
        transform = _make_transform(grid, grid, None)
        for spin in (1, -1):
            degrees = sorted({1, bandlimit // 2, bandlimit - 2, bandlimit - 1})
            precomps = _precomputes(bandlimit, "gl", spin, forward=False)
            for degree in degrees:
                if not abs(spin) <= degree < bandlimit:
                    continue
                orders = {
                    0,
                    1,
                    -1,
                    degree // 2,
                    -(degree // 2),
                    degree,
                    -degree,
                    bandlimit - 2,
                    -(bandlimit - 2),
                    bandlimit - 1,
                    -(bandlimit - 1),
                }
                for order in sorted(value for value in orders if abs(value) <= degree):
                    coefficients = (
                        jnp.zeros((bandlimit, 2 * bandlimit - 1), dtype=jnp.complex128)
                        .at[degree, bandlimit - 1 + order]
                        .set(0.7 - 0.3j)
                    )
                    native = s2fft.inverse_jax(
                        coefficients,
                        bandlimit,
                        spin=spin,
                        sampling="gl",
                        reality=False,
                        precomps=precomps,
                        spmd=False,
                    )
                    centered = jnp.fft.fftshift(
                        jnp.fft.fft(native, axis=-1, norm="forward"), axes=-1
                    )
                    padded = jnp.concatenate(
                        (jnp.zeros_like(centered[..., :1]), centered), axis=-1
                    )
                    expected_field = jnp.fft.ifft(
                        jnp.fft.ifftshift(padded, axes=-1),
                        axis=-1,
                        norm="forward",
                    )
                    fixture = f"analytic_l{degree}_m{order}"
                    for method in METHODS:
                        select_method(method)
                        analyzed = _spin_analysis(expected_field, transform, spin)
                        synthesized = _spin_synthesis(coefficients, transform, spin)
                        rows.extend(
                            (
                                _row(
                                    method,
                                    bandlimit,
                                    f"{spin:+d}",
                                    fixture,
                                    "analytic_spin_forward_coefficients_vs_native_s2fft",
                                    analyzed,
                                    coefficients,
                                ),
                                _row(
                                    method,
                                    bandlimit,
                                    f"{spin:+d}",
                                    fixture,
                                    "analytic_spin_inverse_field_vs_native_s2fft",
                                    synthesized,
                                    expected_field,
                                ),
                            )
                        )
    return rows


def _nyquist_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for bandlimit in (4, 8, 16, 32, 64, 128):
        grid = sg.gaussian_grid(
            bandlimit, 2 * bandlimit, latitude_order="descending", lon0=0.0
        )
        transform = _make_transform(grid, grid, None)
        nyquist = np.broadcast_to(
            (-1.0) ** np.arange(2 * bandlimit), (bandlimit, 2 * bandlimit)
        ).copy()
        for method in METHODS:
            select_method(method)
            coefficients = _scalar_analysis(jnp.asarray(nyquist), transform)
            rows.append(
                _row(
                    method,
                    bandlimit,
                    "0",
                    "pure_longitude_nyquist",
                    "forward_projection_coefficients",
                    coefficients,
                    np.zeros((bandlimit, 2 * bandlimit - 1), dtype=np.complex128),
                )
            )
            complex_nyquist = nyquist.astype(np.complex128) * (0.75 - 0.5j)
            for spin in (1, -1):
                spin_coefficients = _spin_analysis(
                    jnp.asarray(complex_nyquist), transform, spin
                )
                rows.append(
                    _row(
                        method,
                        bandlimit,
                        f"{spin:+d}",
                        "pure_complex_longitude_nyquist",
                        "forward_projection_coefficients",
                        spin_coefficients,
                        np.zeros((bandlimit, 2 * bandlimit - 1), dtype=np.complex128),
                    )
                )
            packed = _random_spectrum(bandlimit, "full_spectrum_random_phase", 8801)
            assert isinstance(packed, np.ndarray)
            values = _scalar_field_from_coefficients(packed, bandlimit, grid)
            valid_coefficients = _scalar_analysis(jnp.asarray(values), transform)
            synthesized = _scalar_synthesis(valid_coefficients, transform)
            spectrum = jnp.fft.fftshift(
                jnp.fft.fft(synthesized, axis=-1, norm="forward"), axes=-1
            )[..., 0]
            rows.append(
                _row(
                    method,
                    bandlimit,
                    "0",
                    "random_valid_spherical_coefficients",
                    "inverse_nyquist_slot",
                    spectrum,
                    np.zeros((bandlimit,), dtype=np.complex128),
                )
            )
            rng = np.random.default_rng(9900 + bandlimit)
            for spin in (1, -1):
                coefficients = rng.standard_normal(
                    (bandlimit, 2 * bandlimit - 1)
                ) + 1j * rng.standard_normal((bandlimit, 2 * bandlimit - 1))
                coefficients[0] = 0.0
                field = _spin_synthesis(jnp.asarray(coefficients), transform, spin)
                spectrum = jnp.fft.fftshift(
                    jnp.fft.fft(field, axis=-1, norm="forward"), axes=-1
                )[..., 0]
                rows.append(
                    _row(
                        method,
                        bandlimit,
                        f"{spin:+d}",
                        "random_valid_spin_coefficients",
                        "inverse_nyquist_slot",
                        spectrum,
                        np.zeros((bandlimit,), dtype=np.complex128),
                    )
                )
    return rows


def _leading_dimension_rows() -> list[dict[str, object]]:
    bandlimit = 8
    grid = _gl_grid_variants(bandlimit)["half_step_ascending"]
    base = np.asarray(scalar_values_for_grid(grid), dtype=np.float64)
    shapes = {
        "single": base,
        "singleton_batch": base[np.newaxis, ...],
        "batch": np.stack((base, 0.5 * base, -base, 2.0 * base)),
        "time_level": np.stack(
            (np.stack((base, 0.5 * base)), np.stack((-base, 2.0 * base)))
        ),
    }
    rows: list[dict[str, object]] = []
    for fixture, values in shapes.items():
        leading_shape = values.shape[:-2]
        dims = tuple(f"d{index}" for index in range(len(leading_shape))) + (
            "lat",
            "lon",
        )
        expected = sg.filter(
            xr.DataArray(
                values,
                dims=dims,
                coords={"lat": grid.latitude, "lon": grid.longitude},
            ),
            "T5",
        ).data
        for method in METHODS:
            select_method(method)
            actual = sgj.filter(jnp.asarray(values), "T5", grid=grid)
            rows.append(
                _row(
                    method,
                    bandlimit,
                    "0",
                    fixture,
                    "leading_dimensions_filter_vs_ducc",
                    actual,
                    expected,
                )
            )
    return rows


def scalar_values_for_grid(grid: Grid) -> np.ndarray:
    latitude = np.deg2rad(grid.latitude)[:, None]
    longitude = np.deg2rad(grid.longitude)[None, :]
    return (
        0.7
        + 0.2 * np.sin(latitude)
        + 0.4 * np.cos(latitude) * np.cos(longitude)
        + 0.1 * np.cos(latitude) ** 2 * np.sin(3.0 * longitude)
    )


def _api_operation_rows() -> list[dict[str, object]]:
    """Compare the shared high-level API surface with DUCC on one GL 2L grid."""
    bandlimit = 8
    grid = _gl_grid_variants(bandlimit)["half_step_ascending"]
    target = sg.gaussian_grid(6, 12, lon0=123.0, latitude_order="descending")
    values = scalar_values_for_grid(grid)
    u_values = np.sin(np.deg2rad(grid.latitude))[:, None] * np.cos(
        np.deg2rad(grid.longitude)[None, :]
    )
    v_values = np.cos(np.deg2rad(grid.latitude))[:, None] * np.sin(
        np.deg2rad(grid.longitude)[None, :]
    )
    u_values = np.broadcast_to(u_values, values.shape).copy()
    v_values = np.broadcast_to(v_values, values.shape).copy()
    scalar = jnp.asarray(values, dtype=jnp.float64)
    u = jnp.asarray(u_values, dtype=jnp.float64)
    v = jnp.asarray(v_values, dtype=jnp.float64)
    scalar_reference = xr.DataArray(
        values,
        dims=("lat", "lon"),
        coords={"lat": grid.latitude, "lon": grid.longitude},
    )
    u_reference = xr.DataArray(
        u_values,
        dims=("lat", "lon"),
        coords={"lat": grid.latitude, "lon": grid.longitude},
        name="u",
    )
    v_reference = xr.DataArray(
        v_values,
        dims=("lat", "lon"),
        coords={"lat": grid.latitude, "lon": grid.longitude},
        name="v",
    )
    rows: list[dict[str, object]] = []

    def add(
        method: str,
        operation: str,
        actual: object,
        expected: object,
        spin: str = "0",
    ) -> None:
        rows.append(
            _row(method, bandlimit, spin, "api_fixture", operation, actual, expected)
        )

    for method in METHODS:
        select_method(method)
        add(
            method,
            "filter",
            sgj.filter(scalar, "T5", grid=grid),
            sg.filter(scalar_reference, "T5").data,
        )
        add(
            method,
            "regrid",
            sgj.regrid(scalar, target, "T5", source_grid=grid),
            sg.regrid(scalar_reference, target, "T5").data,
        )
        vector_regrid = sgj.regrid_vector(u, v, target, "T5", source_grid=grid)
        expected_vector_regrid = sg.regrid_vector(
            u_reference, v_reference, target, "T5"
        )
        add(
            method,
            "regrid_vector_u",
            vector_regrid[0],
            expected_vector_regrid.u.data,
            "±1",
        )
        add(
            method,
            "regrid_vector_v",
            vector_regrid[1],
            expected_vector_regrid.v.data,
            "±1",
        )

        expected_gradient = sg.gradient(scalar_reference)
        actual_gradient = sgj.gradient(scalar, grid=grid)
        add(
            method,
            "gradient_eastward",
            actual_gradient[0],
            expected_gradient.gradient_eastward.data,
        )
        add(
            method,
            "gradient_northward",
            actual_gradient[1],
            expected_gradient.gradient_northward.data,
        )
        add(
            method,
            "inverse_gradient",
            sgj.inverse_gradient(*actual_gradient, grid=grid),
            sg.inverse_gradient(
                expected_gradient.gradient_eastward,
                expected_gradient.gradient_northward,
            ).data,
        )
        add(
            method,
            "laplacian",
            sgj.laplacian(scalar, grid=grid),
            sg.laplacian(scalar_reference).data,
        )
        add(
            method,
            "inverse_laplacian",
            sgj.inverse_laplacian(scalar, grid=grid),
            sg.inverse_laplacian(scalar_reference).data,
        )

        expected_vector_laplacian = sg.vector_laplacian(u_reference, v_reference)
        actual_vector_laplacian = sgj.vector_laplacian(u, v, grid=grid)
        add(
            method,
            "vector_laplacian_u",
            actual_vector_laplacian[0],
            expected_vector_laplacian.u.data,
            "±1",
        )
        add(
            method,
            "vector_laplacian_v",
            actual_vector_laplacian[1],
            expected_vector_laplacian.v.data,
            "±1",
        )
        expected_inverse_vector_laplacian = sg.inverse_vector_laplacian(
            u_reference, v_reference
        )
        actual_inverse_vector_laplacian = sgj.inverse_vector_laplacian(u, v, grid=grid)
        add(
            method,
            "inverse_vector_laplacian_u",
            actual_inverse_vector_laplacian[0],
            expected_inverse_vector_laplacian.u.data,
            "±1",
        )
        add(
            method,
            "inverse_vector_laplacian_v",
            actual_inverse_vector_laplacian[1],
            expected_inverse_vector_laplacian.v.data,
            "±1",
        )
        expected_vorticity = sg.vorticity(u_reference, v_reference)
        expected_divergence = sg.divergence(u_reference, v_reference)
        actual_vorticity, actual_divergence = sgj.kinematics(u, v, grid=grid)
        add(
            method,
            "vorticity",
            sgj.vorticity(u, v, grid=grid),
            expected_vorticity.data,
            "±1",
        )
        add(
            method,
            "divergence",
            sgj.divergence(u, v, grid=grid),
            expected_divergence.data,
            "±1",
        )
        add(
            method,
            "kinematics_vorticity",
            actual_vorticity,
            expected_vorticity.data,
            "±1",
        )
        add(
            method,
            "kinematics_divergence",
            actual_divergence,
            expected_divergence.data,
            "±1",
        )

        expected_potentials = sg.potentials(u_reference, v_reference)
        actual_potentials = sgj.potentials(u, v, grid=grid)
        add(
            method,
            "potentials_streamfunction",
            actual_potentials[0],
            expected_potentials.strf.data,
            "±1",
        )
        add(
            method,
            "potentials_velocity_potential",
            actual_potentials[1],
            expected_potentials.vp.data,
            "±1",
        )
        add(
            method,
            "streamfunction",
            sgj.streamfunction(u, v, grid=grid),
            expected_potentials.strf.data,
            "±1",
        )
        add(
            method,
            "velocity_potential",
            sgj.velocity_potential(u, v, grid=grid),
            expected_potentials.vp.data,
            "±1",
        )

        expected_helmholtz = sg.helmholtz(u_reference, v_reference)
        actual_helmholtz = sgj.helmholtz(u, v, grid=grid)
        for index, name in enumerate(
            ("u_divergent", "v_divergent", "u_rotational", "v_rotational")
        ):
            add(
                method,
                f"helmholtz_{name}",
                actual_helmholtz[index],
                expected_helmholtz[name].data,
                "±1",
            )

        for source, field in (
            ("vorticity", expected_vorticity),
            ("streamfunction", expected_potentials.strf),
        ):
            actual = sgj.rotational_wind(
                jnp.asarray(field.data), grid=grid, source=source
            )
            expected = sg.rotational_wind(field, source=source)
            add(
                method,
                f"rotational_wind_{source}_u",
                actual[0],
                expected.u_rotational.data,
                "±1",
            )
            add(
                method,
                f"rotational_wind_{source}_v",
                actual[1],
                expected.v_rotational.data,
                "±1",
            )
        for source, field in (
            ("divergence", expected_divergence),
            ("velocity_potential", expected_potentials.vp),
        ):
            actual = sgj.divergent_wind(
                jnp.asarray(field.data), grid=grid, source=source
            )
            expected = sg.divergent_wind(field, source=source)
            add(
                method,
                f"divergent_wind_{source}_u",
                actual[0],
                expected.u_divergent.data,
                "±1",
            )
            add(
                method,
                f"divergent_wind_{source}_v",
                actual[1],
                expected.v_divergent.data,
                "±1",
            )
        for first, second, source in (
            (expected_vorticity, expected_divergence, "vorticity_divergence"),
            (expected_potentials.strf, expected_potentials.vp, "potentials"),
        ):
            actual = sgj.wind(
                jnp.asarray(first.data),
                jnp.asarray(second.data),
                grid=grid,
                source=source,
            )
            expected = sg.wind(first, second, source=source)
            add(method, f"wind_{source}_u", actual[0], expected.u.data, "±1")
            add(method, f"wind_{source}_v", actual[1], expected.v.data, "±1")
    return rows


def _one_transformation_candidate(
    method: str,
    transform: _JaxTransform,
    grid: Grid,
    values: Array,
    tangent: Array,
    coefficients: Array,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, Array]]:
    bandlimit = grid.nlat
    select_method(method)

    def forward(field: Array) -> Array:
        return _scalar_analysis(field, transform)

    def inverse(coeffs: Array) -> Array:
        return _scalar_synthesis(coeffs, transform)

    rows: list[dict[str, object]] = []
    status: dict[str, object] = {}
    compiled = jit(forward)(values)
    batch = jnp.stack((values, 1.5 * values))
    vmapped = vmap(jit(forward))(batch)
    jit_vmapped = jit(vmap(forward))(batch)
    rows.append(
        _row(method, bandlimit, "0", "transform", "jit_forward", compiled, compiled)
    )
    rows.append(
        _row(
            method,
            bandlimit,
            "0",
            "transform",
            "vmap_jit_forward",
            vmapped,
            jit_vmapped,
        )
    )

    for name, function, primal, direction in (
        ("forward", forward, values, tangent),
        ("inverse", inverse, coefficients, jnp.ones_like(coefficients)),
    ):
        try:
            _, derivative = jvp(function, (primal,), (direction,))
        except TypeError as error:
            status[f"{method}_{name}_jvp"] = {
                "status": "unsupported",
                "reason": str(error).splitlines()[0],
            }
        else:
            status[f"{method}_{name}_jvp"] = {"status": "passed"}
            rows.append(
                _row(
                    method,
                    bandlimit,
                    "0",
                    "transform",
                    f"{name}_jvp_vs_linear_action",
                    derivative,
                    function(direction),
                )
            )

    forward_gradient = grad(
        lambda field: jnp.real(jnp.vdot(forward(field), forward(field)))
    )(values)
    inverse_gradient = grad(
        lambda coeffs: jnp.real(jnp.vdot(inverse(coeffs), inverse(coeffs)))
    )(coefficients)
    composition_gradient = grad(
        lambda field: jnp.sum(sgj.filter(field, "T5", grid=grid) ** 2)
    )(values)
    q = jnp.asarray(jnp.sin(values) + 1j * jnp.cos(values), dtype=jnp.complex128)

    def spin_forward(field: Array) -> Array:
        return _spin_analysis(field, transform, 1)

    spin_gradient = grad(
        lambda field: jnp.real(jnp.vdot(spin_forward(field), spin_forward(field)))
    )(q)
    vector_gradient = grad(
        lambda u: jnp.sum(sgj.kinematics(u, values, grid=grid)[0] ** 2)
    )(values)
    gradients = cast(
        dict[str, Array],
        {
            "forward": forward_gradient,
            "inverse": inverse_gradient,
            "composition": composition_gradient,
            "spin": spin_gradient,
            "vector": vector_gradient,
        },
    )
    status[f"{method}_grad"] = {
        "forward_finite": bool(jnp.isfinite(forward_gradient).all()),
        "inverse_finite": bool(jnp.isfinite(inverse_gradient).all()),
        "composition_finite": bool(jnp.isfinite(composition_gradient).all()),
        "spin_finite": bool(jnp.isfinite(spin_gradient).all()),
        "vector_finite": bool(jnp.isfinite(vector_gradient).all()),
    }
    return rows, status, gradients


def _transformation_rows() -> tuple[list[dict[str, object]], dict[str, object]]:
    bandlimit = 8
    grid = _gl_grid_variants(bandlimit)["half_step_ascending"]
    transform = _make_transform(grid, grid, None)
    values = jnp.asarray(scalar_values_for_grid(grid), dtype=jnp.float64)
    tangent = jnp.sin(jnp.arange(values.size, dtype=jnp.float64)).reshape(values.shape)
    coefficients_reference = _analyze_scalar_reference(
        np.asarray(values), bandlimit, grid
    )
    coefficients = jnp.asarray(coefficients_reference)
    rows: list[dict[str, object]] = []
    status: dict[str, object] = {}
    gradients_by_method: dict[str, dict[str, Array]] = {}
    for method in METHODS:
        candidate_rows, candidate_status, gradients = _one_transformation_candidate(
            method, transform, grid, values, tangent, coefficients
        )
        rows.extend(candidate_rows)
        status.update(candidate_status)
        gradients_by_method[method] = gradients
    for name in ("forward", "inverse", "composition", "spin", "vector"):
        rows.append(
            _row(
                "public_resample_vs_internal_ftm",
                bandlimit,
                "+1" if name == "spin" else "±1" if name == "vector" else "0",
                "transform",
                f"{name}_gradient_candidate_difference",
                gradients_by_method["public_resample"][name],
                gradients_by_method["internal_ftm"][name],
            )
        )
    return rows, status


def run(sizes: tuple[int, ...], output: Path) -> dict[str, object]:
    if not bool(config.read("jax_enable_x64")):
        config.update("jax_enable_x64", True)
    rows = []
    rows.extend(_scalar_matrix_rows(sizes))
    rows.extend(_vector_matrix_rows(sizes))
    rows.extend(_analytic_spin_rows(sizes))
    rows.extend(_nyquist_rows())
    rows.extend(_leading_dimension_rows())
    rows.extend(_api_operation_rows())
    transformation_rows, transformation_status = _transformation_rows()
    rows.extend(transformation_rows)
    document = {
        "metadata": _metadata(),
        "candidate_methods": list(METHODS),
        "sizes": list(sizes),
        "transformation_status": transformation_status,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=DEFAULT_SIZES,
        help="bandlimits for analytic and random spectral fixtures",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks/jax_gl_2l/results/validation.json",
    )
    args = parser.parse_args()
    document = run(tuple(args.sizes), args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rows": len(cast(list[dict[str, object]], document["rows"])),
                "metadata": document["metadata"],
                "transformation_status": document["transformation_status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
