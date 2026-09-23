# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark both experimental JAX paths on atmospheric GL ``(L, 2L)`` grids."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from statistics import median

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
from _experiment import METHODS, select_method
from jax.stages import Lowered

import spharmgrid as sg
import spharmgrid.jax as sgj
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
DEFAULT_SIZES = (8, 16, 32, 64, 128, 256, 512)


def _cpu_model() -> str | None:
    try:
        text = Path("/proc/cpuinfo").read_text()
    except OSError:
        return platform.processor() or None
    match = re.search(r"^model name\s*:\s*(.+)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else (platform.processor() or None)


def _environment() -> dict[str, object]:
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
        "cpu_model": _cpu_model(),
        "spharmgrid_commit": commit,
        "s2fft_version": distribution.version,
        "s2fft_direct_url": direct_url,
        "jax_version": jax.__version__,
        "jaxlib_version": importlib.metadata.version("jaxlib"),
        "jax_x64_enabled": bool(jax.config.read("jax_enable_x64")),
        "jax_backend": jax.default_backend(),
        "jax_devices": [
            {
                "platform": device.platform,
                "device_kind": getattr(device, "device_kind", None),
                "id": device.id,
            }
            for device in jax.devices()
        ],
    }


def _block(value: object) -> None:
    jax.block_until_ready(value)


def _repetitions(bandlimit: int) -> int:
    if bandlimit <= 32:
        return 20
    if bandlimit <= 64:
        return 12
    if bandlimit <= 128:
        return 8
    if bandlimit <= 256:
        return 5
    return 3


def _quantiles(samples: list[float]) -> dict[str, float]:
    values = np.asarray(samples, dtype=np.float64)
    return {
        "median_ms": float(median(samples)),
        "minimum_ms": float(np.min(values)),
        "p25_ms": float(np.quantile(values, 0.25)),
        "p75_ms": float(np.quantile(values, 0.75)),
        "maximum_ms": float(np.max(values)),
    }


def _memory_analysis(executable: object) -> dict[str, int] | None:
    method = getattr(executable, "memory_analysis", None)
    if not callable(method):
        return None
    stats = method()
    if stats is None:
        return None
    names = (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "alias_size_in_bytes",
        "temp_size_in_bytes",
        "host_temp_size_in_bytes",
    )
    return {
        name: int(value)
        for name in names
        if (value := getattr(stats, name, None)) is not None
    }


def _stablehlo_summary(lowered: Lowered) -> dict[str, int]:
    module = lowered.compiler_ir(dialect="stablehlo")
    text = str(module)
    return {
        "stablehlo_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "stablehlo_characters": len(text),
        "stablehlo_lines": text.count("\n"),
        "stablehlo_operations": len(re.findall(r"\bstablehlo\.[a-zA-Z0-9_]+", text)),
    }


def _measure_compiled(
    function: Callable[..., object],
    args: tuple[jax.Array, ...],
    *,
    method: str,
    operation: str,
    bandlimit: int,
    batch: int,
    repeat_count: int,
    dtype: str,
    extra: dict[str, object],
) -> dict[str, object]:
    # S2FFT's public transform and the internal candidate call shared jitted
    # latitude functions. Clear JAX's executable cache before each case so the
    # first candidate cannot precompile work later attributed to the second.
    jax.clear_caches()
    jitted = jax.jit(function)
    start = time.perf_counter()
    traced = jitted.trace(*args)
    trace_ms = (time.perf_counter() - start) * 1000.0
    start = time.perf_counter()
    lowered = traced.lower()
    lowering_ms = (time.perf_counter() - start) * 1000.0
    stablehlo = _stablehlo_summary(lowered)
    start = time.perf_counter()
    executable = lowered.compile()
    compile_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    first_output = executable(*args)
    _block(first_output)
    first_call_ms = (time.perf_counter() - start) * 1000.0

    samples: list[float] = []
    for _ in range(repeat_count):
        start = time.perf_counter()
        output = executable(*args)
        _block(output)
        samples.append((time.perf_counter() - start) * 1000.0)

    return {
        "candidate": method,
        "L": bandlimit,
        "operation": operation,
        "batch": batch,
        "dtype": dtype,
        "device": jax.default_backend(),
        "trace_ms": trace_ms,
        "lowering_ms": lowering_ms,
        "compile_ms": compile_ms,
        "first_call_after_compile_ms": first_call_ms,
        "repetitions": repeat_count,
        **_quantiles(samples),
        "memory_analysis": _memory_analysis(executable),
        **stablehlo,
        **extra,
    }


def _batch(values: np.ndarray, batch_size: int) -> np.ndarray:
    if batch_size == 1:
        return values
    factors = np.linspace(0.75, 1.25, batch_size, dtype=np.float64)
    return factors.reshape((batch_size,) + (1,) * values.ndim) * values


def _prepare_precomputes(bandlimit: int) -> tuple[list[dict[str, object]], float]:
    rows: list[dict[str, object]] = []
    _precomputes.cache_clear()
    total_ms = 0.0
    for spin in (0, 1, -1):
        for forward in (True, False):
            start = time.perf_counter()
            values = _precomputes(bandlimit, "gl", spin, forward=forward)
            for value in values:
                value.block_until_ready()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            total_ms += elapsed_ms
            rows.append(
                {
                    "L": bandlimit,
                    "spin": spin,
                    "direction": "forward" if forward else "inverse",
                    "precompute_generation_ms": elapsed_ms,
                    "array_shapes": [list(value.shape) for value in values],
                    "array_bytes": sum(
                        value.size * value.dtype.itemsize for value in values
                    ),
                }
            )
    return rows, total_ms


def _resampling_intermediates(
    method: str, operation: str, batch: int, bandlimit: int
) -> dict[str, int]:
    """Count the extra native-grid physical arrays required by candidate A."""
    if method != "public_resample":
        return {
            "forward_resample_physical_bytes": 0,
            "inverse_resample_physical_bytes": 0,
        }
    itemsize = bandlimit * (2 * bandlimit - 1) * batch
    forward_components, forward_itemsize = 0, 0
    inverse_components, inverse_itemsize = 0, 0
    if operation == "forward_scalar":
        forward_components, forward_itemsize = 1, 8
    elif operation == "forward_spin_plus1":
        forward_components, forward_itemsize = 1, 16
    elif operation == "inverse_scalar":
        inverse_components, inverse_itemsize = 1, 8
    elif operation == "inverse_spin_plus1":
        inverse_components, inverse_itemsize = 1, 16
    elif operation == "vector_analysis":
        forward_components, forward_itemsize = 2, 16
    elif operation == "vector_synthesis":
        inverse_components, inverse_itemsize = 2, 16
    elif operation == "filter_round_trip":
        forward_components, forward_itemsize = 1, 8
        inverse_components, inverse_itemsize = 1, 8
    elif operation == "kinematics":
        forward_components, forward_itemsize = 2, 16
        inverse_components, inverse_itemsize = 2, 8
    return {
        "forward_resample_physical_bytes": (
            itemsize * forward_components * forward_itemsize
        ),
        "inverse_resample_physical_bytes": (
            itemsize * inverse_components * inverse_itemsize
        ),
    }


def _operation_functions(
    grid: sg.Grid,
    transform: _JaxTransform,
    spin_coefficients: jax.Array,
    vector_coefficients: jax.Array,
    truncation: str,
) -> dict[str, tuple[Callable[..., object], tuple[jax.Array, ...], str]]:
    bandlimit = grid.nlat
    rng = np.random.default_rng(5000 + bandlimit)
    scalar = jnp.asarray(
        rng.standard_normal((bandlimit, 2 * bandlimit)), dtype=jnp.float64
    )
    spin_field = jnp.asarray(
        rng.standard_normal((bandlimit, 2 * bandlimit))
        + 1j * rng.standard_normal((bandlimit, 2 * bandlimit)),
        dtype=jnp.complex128,
    )
    eastward = jnp.asarray(
        rng.standard_normal((bandlimit, 2 * bandlimit)), dtype=jnp.float64
    )
    northward = jnp.asarray(
        rng.standard_normal((bandlimit, 2 * bandlimit)), dtype=jnp.float64
    )
    functions: dict[str, tuple[Callable[..., object], tuple[jax.Array, ...], str]] = {
        "forward_scalar": (
            lambda field: _scalar_analysis(field, transform),
            (scalar,),
            "float64",
        ),
        "inverse_scalar": (
            lambda coeff: _scalar_synthesis(coeff, transform),
            (vector_coefficients[0],),
            "complex128",
        ),
        "forward_spin_plus1": (
            lambda field: _spin_analysis(field, transform, 1),
            (spin_field,),
            "complex128",
        ),
        "inverse_spin_plus1": (
            lambda coeff: _spin_synthesis(coeff, transform, 1),
            (spin_coefficients,),
            "complex128",
        ),
        "vector_analysis": (
            lambda u, v: _vector_analysis(u, v, transform),
            (eastward, northward),
            "float64",
        ),
        "vector_synthesis": (
            lambda coeff: _vector_synthesis(coeff, transform),
            (vector_coefficients,),
            "complex128",
        ),
        "filter_round_trip": (
            lambda field: sgj.filter(field, truncation, grid=grid),
            (scalar,),
            "float64",
        ),
        "kinematics": (
            lambda u, v: sgj.kinematics(u, v, grid=grid),
            (eastward, northward),
            "float64",
        ),
    }
    return functions


def _measure_size(
    bandlimit: int,
    *,
    batch_sizes: tuple[int, ...],
    include_all_batches: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    start = time.perf_counter()
    grid = sg.gaussian_grid(
        bandlimit,
        2 * bandlimit,
        lon0=0.0,
        latitude_order="descending",
    )
    transform = _make_transform(grid, grid, None)
    rng = np.random.default_rng(8000 + bandlimit)
    scalar = rng.standard_normal((bandlimit, 2 * bandlimit))
    spin_host = rng.standard_normal(
        (bandlimit, 2 * bandlimit - 1)
    ) + 1j * rng.standard_normal((bandlimit, 2 * bandlimit - 1))
    vector_host = rng.standard_normal(
        (2, bandlimit, 2 * bandlimit - 1)
    ) + 1j * rng.standard_normal((2, bandlimit, 2 * bandlimit - 1))
    setup_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    _device_inputs = (
        jax.device_put(scalar),
        jax.device_put(spin_host),
        jax.device_put(vector_host),
    )
    _block(_device_inputs)
    device_transfer_ms = (time.perf_counter() - start) * 1000.0

    try:
        precompute_rows, precompute_ms = _prepare_precomputes(bandlimit)
    except Exception as error:
        precompute_ms = float("nan")
        precompute_rows = [
            {
                "L": bandlimit,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        ]
    truncation = f"T{min(bandlimit - 1, 32)}"
    operation_setup_start = time.perf_counter()
    all_operations = _operation_functions(
        grid,
        transform,
        jnp.asarray(spin_host, dtype=jnp.complex128),
        jnp.asarray(vector_host, dtype=jnp.complex128),
        truncation,
    )
    operation_setup_ms = (time.perf_counter() - operation_setup_start) * 1000.0
    rows: list[dict[str, object]] = []
    for method in METHODS:
        select_method(method)
        for operation, (function, base_args, dtype) in all_operations.items():
            batches = (1,)
            if include_all_batches and operation in {
                "forward_scalar",
                "inverse_scalar",
            }:
                batches = batch_sizes
            for batch_size in batches:
                if operation in {"forward_scalar", "filter_round_trip"}:
                    args = (_batch(np.asarray(base_args[0]), batch_size),)
                    args = (jnp.asarray(args[0], dtype=jnp.float64),)
                elif operation in {
                    "inverse_scalar",
                    "forward_spin_plus1",
                    "inverse_spin_plus1",
                }:
                    args = (_batch(np.asarray(base_args[0]), batch_size),)
                    args = (jnp.asarray(args[0], dtype=jnp.complex128),)
                elif operation in {"vector_analysis", "kinematics"}:
                    args = tuple(
                        jnp.asarray(
                            _batch(np.asarray(value), batch_size), dtype=jnp.float64
                        )
                        for value in base_args
                    )
                else:
                    args = tuple(
                        jnp.asarray(
                            _batch(np.asarray(value), batch_size), dtype=jnp.complex128
                        )
                        for value in base_args
                    )

                extra = {
                    "python_setup_ms_for_size": setup_ms,
                    "device_transfer_ms_for_size": device_transfer_ms,
                    "operation_setup_ms_for_size": operation_setup_ms,
                    "precompute_generation_ms_for_size": (
                        precompute_ms if np.isfinite(precompute_ms) else None
                    ),
                    "steady_repetitions": _repetitions(bandlimit),
                    **_resampling_intermediates(
                        method, operation, batch_size, bandlimit
                    ),
                }
                try:
                    measured = _measure_compiled(
                        function,
                        args,
                        method=method,
                        operation=operation,
                        bandlimit=bandlimit,
                        batch=batch_size,
                        repeat_count=_repetitions(bandlimit),
                        dtype=dtype,
                        extra=extra,
                    )
                except Exception as error:
                    measured = {
                        "candidate": method,
                        "L": bandlimit,
                        "operation": operation,
                        "batch": batch_size,
                        "dtype": dtype,
                        "device": jax.default_backend(),
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        **extra,
                    }
                rows.append(measured)
    setup_row = {
        "L": bandlimit,
        "python_setup_ms": setup_ms,
        "device_transfer_ms": device_transfer_ms,
        "operation_setup_ms": operation_setup_ms,
        "precompute_generation_ms": (
            precompute_ms if np.isfinite(precompute_ms) else None
        ),
        "precompute_status": precompute_rows[0].get("status", "passed"),
        "physical_input_bytes_scalar_float64": bandlimit * 2 * bandlimit * 8,
        "physical_input_bytes_spin_complex128": bandlimit * 2 * bandlimit * 16,
        "ftm_bytes_complex128": bandlimit * (2 * bandlimit - 1) * 16,
        "public_resample_intermediate_bytes_scalar": bandlimit
        * (2 * bandlimit - 1)
        * 8,
        "public_resample_intermediate_bytes_spin": bandlimit * (2 * bandlimit - 1) * 16,
        "internal_ftm_extra_physical_intermediate_bytes": 0,
        "public_resample_vector_analysis_intermediate_bytes": (
            2 * bandlimit * (2 * bandlimit - 1) * 16
        ),
        "public_resample_vector_synthesis_intermediate_bytes": (
            2 * bandlimit * (2 * bandlimit - 1) * 16
        ),
    }
    return rows, precompute_rows, setup_row


def run(sizes: tuple[int, ...], output: Path) -> dict[str, object]:
    if not bool(jax.config.read("jax_enable_x64")):
        jax.config.update("jax_enable_x64", True)
    performance_rows: list[dict[str, object]] = []
    setup_rows: list[dict[str, object]] = []
    precompute_rows: list[dict[str, object]] = []
    for bandlimit in sizes:
        if bandlimit <= 64:
            batch_sizes = (1, 4, 16)
            include_all_batches = True
        elif bandlimit <= 128:
            batch_sizes = (1, 4)
            include_all_batches = True
        else:
            batch_sizes = (1,)
            include_all_batches = False
        rows, precomputes, setup = _measure_size(
            bandlimit,
            batch_sizes=batch_sizes,
            include_all_batches=include_all_batches,
        )
        performance_rows.extend(rows)
        precompute_rows.extend(precomputes)
        setup_rows.append(setup)
        print(
            f"completed L={bandlimit}: {len(rows)} timing rows on "
            f"{jax.default_backend()}"
        )
    document = {
        "environment": _environment(),
        "sizes": list(sizes),
        "methods": list(METHODS),
        "methodology": {
            "steady_state_repetitions": (
                "20 (L<=32), 12 (L<=64), 8 (L<=128), 5 (L<=256), 3 (larger)"
            ),
            "synchronization": "block_until_ready on every measured output",
            "compile_excluded_from_steady_state": True,
            "memory": (
                "compiled memory_analysis where exposed; StableHLO size/op count "
                "and array-shape byte counts otherwise"
            ),
        },
        "setup": setup_rows,
        "precomputes": precompute_rows,
        "performance": performance_rows,
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
        help="bandlimits for the benchmark matrix",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks/jax_gl_2l/results/performance.json",
    )
    args = parser.parse_args()
    document = run(tuple(args.sizes), args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "performance_rows": len(document["performance"]),
                "environment": document["environment"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
