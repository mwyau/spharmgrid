# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare eager and Dask Xarray transform costs across spharmgrid revisions.

Run this same file from identical Python environments that install different
worktrees of spharmgrid. The JSON output records graph construction separately
from Dask computation and stores arrays for cross-revision numerical checks.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from collections.abc import Callable
from importlib.metadata import distributions
from pathlib import Path
from typing import Any

import dask
import ducc0
import numpy as np
import xarray as xr

import spharmgrid as sg

NTHREADS = 1
DASK_WORKERS = 2
NREPEATS = 5
ATOL = 3.0e-14

GRID_SHAPE = (64, 128)
GRID_PAIRS = (("gl", "cc"), ("cc", "gl"), ("f1", "gl"), ("gl", "f1"))
SHAPES = {
    "single": ((), ("lat", "lon")),
    "eight": ((2, 4), ("time", "level", "lat", "lon")),
    "sixty_four": ((8, 8), ("time", "level", "lat", "lon")),
    "sixty_four_interleaved": ((8, 8), ("time", "lat", "level", "lon")),
}


def _grid(kind: str) -> sg.Grid:
    constructor = {
        "gl": sg.gaussian_grid,
        "cc": sg.clenshaw_curtis_grid,
        "f1": sg.fejer1_grid,
    }[kind]
    return constructor(*GRID_SHAPE)


def _coordinates(grid: sg.Grid, shape_name: str) -> dict[str, Any]:
    coordinates: dict[str, Any] = {"lat": grid.latitude, "lon": grid.longitude}
    leading_shape, _ = SHAPES[shape_name]
    if leading_shape:
        coordinates["time"] = np.arange(leading_shape[0])
        coordinates["level"] = np.arange(leading_shape[1])
    return coordinates


def _data_array(
    values: np.ndarray[Any, np.dtype[np.float64]],
    grid: sg.Grid,
    shape_name: str,
    name: str,
) -> xr.DataArray:
    leading_shape, leading_names = SHAPES[shape_name]
    if shape_name == "sixty_four_interleaved":
        dims = ("time", "lat", "level", "lon")
        shape = (leading_shape[0], grid.nlat, leading_shape[1], grid.nlon)
        values = values.reshape(shape)
    else:
        dims = leading_names
        shape = (*leading_shape, grid.nlat, grid.nlon)
        values = values.reshape(shape)
    return xr.DataArray(
        values,
        dims=dims,
        coords=_coordinates(grid, shape_name),
        name=name,
    )


def _seed(*parts: str) -> int:
    # Stable across Python processes unlike the built-in hash().
    state = 17
    for part in parts:
        for byte in part.encode():
            state = (state * 131 + byte) % (2**32)
    return state


def _field_pair(grid: sg.Grid, shape_name: str) -> tuple[xr.DataArray, xr.DataArray]:
    leading_shape, _ = SHAPES[shape_name]
    nfield = int(np.prod(leading_shape)) if leading_shape else 1
    shape = (nfield, grid.nlat, grid.nlon)
    rng = np.random.default_rng(_seed(grid.kind, shape_name))
    u_values = rng.standard_normal(shape)
    v_values = rng.standard_normal(shape)
    return (
        _data_array(u_values, grid, shape_name, "u"),
        _data_array(v_values, grid, shape_name, "v"),
    )


def _chunk_field(field: xr.DataArray, mode: str, shape_name: str) -> xr.DataArray:
    leading_chunks = {
        "time": 1 if shape_name == "eight" else 2,
        "level": 2,
    }
    chunks: dict[str, int] = {
        dim: min(size, leading_chunks.get(dim, size))
        for dim, size in field.sizes.items()
    }
    if mode == "dask_split":
        chunks["lat"] = field.sizes["lat"] // 2
        chunks["lon"] = field.sizes["lon"] // 2
    else:
        chunks["lat"] = -1
        chunks["lon"] = -1
    return field.chunk(chunks)


def _cases() -> list[tuple[str, str, str, str, str]]:
    cases: list[tuple[str, str, str, str, str]] = []
    for kind in ("gl", "cc", "f1"):
        for shape_name in SHAPES:
            for mode in ("eager", "dask"):
                cases.append(("filter", kind, kind, shape_name, mode))
        cases.append(("filter", kind, kind, "eight", "dask_split"))
        for shape_name in SHAPES:
            for mode in ("eager", "dask"):
                cases.append(("analyze_synthesize", kind, kind, shape_name, mode))
        cases.append(("analyze_synthesize", kind, kind, "eight", "dask_split"))
    for source_kind, target_kind in GRID_PAIRS:
        for shape_name in SHAPES:
            for mode in ("eager", "dask"):
                cases.append(("regrid", source_kind, target_kind, shape_name, mode))
        cases.append(("regrid", source_kind, target_kind, "eight", "dask_split"))
        for shape_name in SHAPES:
            for mode in ("eager", "dask"):
                cases.append(
                    ("regrid_vector", source_kind, target_kind, shape_name, mode)
                )
        cases.append(("regrid_vector", source_kind, target_kind, "eight", "dask_split"))
    return cases


def _make_operation(
    operation: str,
    source: sg.Grid,
    target: sg.Grid,
) -> Callable[[xr.DataArray, xr.DataArray | None], xr.DataArray | xr.Dataset]:
    if operation == "filter":
        return lambda field, _: sg.filter(field, "T31", sht_threads=NTHREADS)
    if operation == "regrid":
        return lambda field, _: sg.regrid(field, target, "T31", sht_threads=NTHREADS)
    if operation == "analyze_synthesize":
        return lambda field, _: sg.analyze(
            field, "T31", sht_threads=NTHREADS
        ).synthesize()
    if operation == "regrid_vector":
        return lambda field, other: sg.regrid_vector(
            field,
            other,
            target,
            "T31",
            eastward="u",
            northward="v",
            sht_threads=NTHREADS,
        )
    raise ValueError(f"unknown operation {operation!r}")


def _materialize(
    result: xr.DataArray | xr.Dataset,
    operation: str,
    *,
    compute: bool,
) -> list[np.ndarray[Any, Any]]:
    if compute:
        with dask.config.set(scheduler="threads", num_workers=DASK_WORKERS):
            result = result.compute()
    if operation == "regrid_vector":
        assert isinstance(result, xr.Dataset)
        return [np.asarray(result[name].data) for name in ("u", "v")]
    assert isinstance(result, xr.DataArray)
    return [np.asarray(result.data)]


def _median_and_mad(values: list[float]) -> tuple[float, float]:
    median = statistics.median(values)
    deviation = statistics.median(abs(value - median) for value in values)
    return median, deviation


def _run_case(
    case: tuple[str, str, str, str, str], repeats: int
) -> tuple[dict[str, Any], list[np.ndarray[Any, Any]]]:
    operation_name, source_kind, target_kind, shape_name, mode = case
    source_grid, target_grid = _grid(source_kind), _grid(target_kind)
    field, other = _field_pair(source_grid, shape_name)
    operation = _make_operation(operation_name, source_grid, target_grid)
    if mode != "eager":
        field = _chunk_field(field, mode, shape_name)
        other = _chunk_field(other, mode, shape_name)

    # Warm the same path before timed runs. Dask compute is deliberately included.
    warm_result = operation(field, other)
    warm_arrays = _materialize(warm_result, operation_name, compute=mode != "eager")
    del warm_arrays

    operation_times: list[float] = []
    graph_times: list[float] = []
    compute_times: list[float] = []
    arrays: list[np.ndarray[Any, Any]] = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = operation(field, other)
        elapsed = time.perf_counter() - started
        if mode == "eager":
            operation_times.append(elapsed)
            arrays = _materialize(result, operation_name, compute=False)
        else:
            graph_times.append(elapsed)
            started = time.perf_counter()
            arrays = _materialize(result, operation_name, compute=True)
            compute_times.append(time.perf_counter() - started)

    times: dict[str, Any] = {}
    if operation_times:
        median, mad = _median_and_mad(operation_times)
        times["elapsed"] = {
            "median_seconds": median,
            "mad_seconds": mad,
            "min_seconds": min(operation_times),
            "max_seconds": max(operation_times),
            "runs_seconds": operation_times,
        }
    else:
        for name, values in (
            ("graph_construction", graph_times),
            ("compute", compute_times),
        ):
            median, mad = _median_and_mad(values)
            times[name] = {
                "median_seconds": median,
                "mad_seconds": mad,
                "min_seconds": min(values),
                "max_seconds": max(values),
                "runs_seconds": values,
            }

    key = "/".join(case)
    result_info = {
        "key": key,
        "operation": operation_name,
        "source_grid": source_kind,
        "target_grid": target_kind,
        "grid_shape": list(GRID_SHAPE),
        "shape_case": shape_name,
        "batch_shape": list(SHAPES[shape_name][0]),
        "dims": list(field.dims),
        "execution": mode,
        "ducc_threads": NTHREADS,
        "dask_workers": DASK_WORKERS if mode != "eager" else None,
        **times,
    }
    return result_info, arrays


def _versions() -> dict[str, Any]:
    package_versions = {
        distribution.metadata["Name"]: distribution.version
        for distribution in distributions()
        if distribution.metadata.get("Name")
    }
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "xarray": xr.__version__,
        "dask": dask.__version__,
        "ducc0": ducc0.__version__,
        "spharmgrid_module": str(Path(sg.__file__).resolve()),
        "packages": dict(
            sorted(package_versions.items(), key=lambda item: item[0].casefold())
        ),
    }


def _load_baseline(
    path: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray[Any, Any]]]:
    report = json.loads(path.read_text())
    archive_path = Path(report["arrays_file"])
    with np.load(archive_path) as archive:
        arrays = {name: archive[name] for name in archive.files}
    return report, arrays


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--repeats", type=int, default=NREPEATS)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.repeats < 5:
        parser.error("--repeats must be at least five")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    baseline_report: dict[str, Any] | None = None
    baseline_arrays: dict[str, np.ndarray[Any, Any]] = {}
    if args.compare:
        baseline_report, baseline_arrays = _load_baseline(args.compare)
        environment_keys = ("python", "numpy", "xarray", "dask", "ducc0", "packages")
        current_versions = _versions()
        differences = {
            name: (baseline_report["versions"].get(name), current_versions[name])
            for name in environment_keys
            if baseline_report["versions"].get(name) != current_versions[name]
        }
        if differences:
            parser.error(f"benchmark environments differ: {differences}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    arrays_path = args.output.with_name(f"{args.output.stem}_arrays.npz")
    report: dict[str, Any] = {
        "versions": _versions(),
        "configuration": {
            "grid_shape": list(GRID_SHAPE),
            "truncation": "T31",
            "ducc_threads": NTHREADS,
            "dask_workers": DASK_WORKERS,
            "repeats": args.repeats,
            "warmups": 1,
            "allclose_rtol": 0.0,
            "allclose_atol": ATOL,
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "BLIS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
        },
        "baseline_versions": baseline_report["versions"] if baseline_report else None,
        "workloads": [],
        "arrays_file": str(arrays_path),
    }
    saved_arrays: dict[str, np.ndarray[Any, Any]] = {}
    failures: list[str] = []
    cases = _cases()
    if args.limit is not None:
        cases = cases[: args.limit]
    report["configuration"]["case_count"] = len(cases)
    if baseline_report is not None:
        config_keys = (
            "grid_shape",
            "truncation",
            "ducc_threads",
            "dask_workers",
            "repeats",
            "warmups",
            "allclose_rtol",
            "allclose_atol",
            "thread_environment",
            "case_count",
        )
        differences = {
            name: (
                baseline_report["configuration"].get(name),
                report["configuration"].get(name),
            )
            for name in config_keys
            if baseline_report["configuration"].get(name)
            != report["configuration"].get(name)
        }
        if differences:
            parser.error(f"benchmark configurations differ: {differences}")
    for index, case in enumerate(cases):
        result_info, outputs = _run_case(case, args.repeats)
        for output_index, output in enumerate(outputs):
            npz_key = f"case_{index}_{output_index}"
            saved_arrays[npz_key] = output
            if baseline_report is not None:
                baseline_key = f"case_{index}_{output_index}"
                reference = baseline_arrays[baseline_key]
                max_abs_error = float(np.max(np.abs(output - reference)))
                matches = bool(np.allclose(output, reference, rtol=0.0, atol=ATOL))
                result_info.setdefault("numerical_comparison", []).append(
                    {
                        "max_abs_error": max_abs_error,
                        "rtol": 0.0,
                        "atol": ATOL,
                        "matches": matches,
                    }
                )
                if not matches:
                    failures.append(f"{result_info['key']} output {output_index}")
        report["workloads"].append(result_info)
        print(json.dumps(result_info, sort_keys=True), flush=True)

    np.savez_compressed(arrays_path, **saved_arrays)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if failures:
        raise SystemExit(
            "outputs exceed existing numerical tolerance: " + ", ".join(failures)
        )


if __name__ == "__main__":
    _main()
