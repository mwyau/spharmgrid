# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Small file-oriented command line interface that delegates I/O to xarray."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from pathlib import Path

import xarray as xr

from .grids import clenshaw_curtis_grid, detect_grid, gaussian_grid
from .kinematics import kinematics, potentials, wind
from .metadata import Quantity, find_variable
from .regrid import regrid
from .spectral import filter

_CLI_INSTALL_MESSAGE = (
    "CLI dependencies are not installed.\n\n"
    'Install them with:\n    pip install "spharmgrid[cli]"'
)
_DASK_CLI_INSTALL_MESSAGE = (
    "Dask is required for transforming CLI commands.\n\n"
    'Install it with:\n    pip install "spharmgrid[cli,dask]"'
)
_DEFAULT_CLI_SHT_THREADS = 4
_GRIB_WINDOWS_PY314_MESSAGE = (
    "GRIB input is not supported on Windows with Python 3.14.\n"
    "Use Python 3.11 to 3.13 for GRIB input."
)


class _CLIDependencyError(RuntimeError):
    """Raised when a file operation needs the optional CLI dependencies."""


@dataclass(frozen=True, slots=True)
class _ExecutionPolicy:
    """Resolved CLI task-worker and DUCC-transform concurrency."""

    workers: int
    sht_threads: int


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``spharmgrid`` command line interface."""
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "info":
            return _info(arguments)
        policy = _resolve_execution_policy(
            workers=arguments.workers,
            sht_threads=arguments.sht_threads,
        )
        with (
            _open_dataset(arguments.input, chunks="auto") as dataset,
            _local_dask_executor(policy.workers),
        ):
            if arguments.command == "filter":
                return _filter(arguments, dataset, policy)
            if arguments.command == "regrid":
                return _regrid(arguments, dataset, policy)
            if arguments.command == "kinematics":
                return _kinematics(arguments, dataset, policy)
            if arguments.command == "potentials":
                return _potentials(arguments, dataset, policy)
            if arguments.command == "wind":
                return _wind(arguments, dataset, policy)
    except _CLIDependencyError as error:
        print(str(error), file=sys.stderr)
        return 2
    except (ImportError, OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    parser.error(f"unknown command {arguments.command!r}")
    return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spharmgrid",
        description=(
            f"spharmgrid {_package_version()} - "
            "Spherical harmonic tools for filtering, regridding, and kinematics. "
            "CLI reads NetCDF, Zarr, and GRIB, and writes NetCDF and Zarr."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s " + _package_version(),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    info = subparsers.add_parser("info", help="show detected GL/CC grid information")
    info.add_argument("input", help="input readable by an installed xarray backend")

    filtered = subparsers.add_parser("filter", help="spectrally filter one variable")
    _input_output_arguments(filtered)
    _spectral_arguments(filtered)
    _execution_arguments(filtered)
    filtered.add_argument(
        "--var", help="variable to filter; required for multi-variable input"
    )

    regridded = subparsers.add_parser("regrid", help="spectrally regrid one variable")
    _input_output_arguments(regridded)
    _spectral_arguments(regridded)
    _execution_arguments(regridded)
    regridded.add_argument(
        "--var", help="variable to regrid; required for multi-variable input"
    )
    regridded.add_argument("--grid", choices=("gl", "cc"), required=True)
    regridded.add_argument("--nlat", type=int, required=True)
    regridded.add_argument("--nlon", type=int, required=True)
    regridded.add_argument("--lon0", type=float, default=0.0)
    regridded.add_argument(
        "--latitude-order", choices=("ascending", "descending"), default="ascending"
    )

    kinematic = subparsers.add_parser(
        "kinematics", help="compute relative vorticity and divergence"
    )
    _input_output_arguments(kinematic)
    _wind_input_arguments(kinematic)
    _execution_arguments(kinematic)
    kinematic.add_argument("--vorticity", default="vo", help="vorticity output name")
    kinematic.add_argument("--divergence", default="d", help="divergence output name")

    potential = subparsers.add_parser(
        "potentials", help="compute streamfunction and velocity potential"
    )
    _input_output_arguments(potential)
    _wind_input_arguments(potential)
    _execution_arguments(potential)
    potential.add_argument(
        "--streamfunction", default="strf", help="streamfunction output name"
    )
    potential.add_argument(
        "--velocity-potential", default="vp", help="velocity-potential output name"
    )

    inverse = subparsers.add_parser(
        "wind", help="reconstruct eastward and northward wind from scalar sources"
    )
    _input_output_arguments(inverse)
    _execution_arguments(inverse)
    inverse.add_argument(
        "--source", choices=("vorticity_divergence", "potentials"), required=True
    )
    inverse.add_argument("--vorticity", help="relative-vorticity source variable")
    inverse.add_argument("--divergence", help="divergence source variable")
    inverse.add_argument("--streamfunction", help="streamfunction source variable")
    inverse.add_argument(
        "--velocity-potential", help="velocity-potential source variable"
    )
    inverse.add_argument("--eastward", default="u", help="eastward-wind output name")
    inverse.add_argument("--northward", default="v", help="northward-wind output name")
    return parser


def _input_output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "input",
        help="input NetCDF, Zarr, or GRIB path handled by an installed xarray backend",
    )
    parser.add_argument(
        "output",
        help="output NetCDF file or .zarr store (GRIB is input-only)",
    )


def _spectral_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--truncation",
        help="Tn, Ta-b, Tnxm, or Rn retained spectral selection",
    )
    parser.add_argument("--lmin", type=int, help="explicit lower retained degree")
    parser.add_argument("--lmax", type=int, help="explicit upper retained degree")
    parser.add_argument("--taper", type=float, help="response at upper retained degree")


def _execution_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workers",
        type=_positive_integer,
        metavar="N",
        help="Dask task workers; inferred from --sht-threads when omitted",
    )
    parser.add_argument(
        "--sht-threads",
        type=_positive_integer,
        metavar="N",
        help="DUCC threads per spherical harmonic transform",
    )


def _positive_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _wind_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--u", help="eastward-wind input variable")
    parser.add_argument("--v", help="northward-wind input variable")


def _resolve_execution_policy(
    *, workers: int | None, sht_threads: int | None
) -> _ExecutionPolicy:
    """Resolve the CLI's Dask-worker and DUCC-thread topology."""
    available_cpus = _available_cpu_count()
    _validate_positive_option(workers, "workers")
    _validate_positive_option(sht_threads, "sht_threads")

    if workers is None:
        resolved_sht_threads = (
            min(_DEFAULT_CLI_SHT_THREADS, available_cpus)
            if sht_threads is None
            else sht_threads
        )
        resolved_workers = _ceil_division(available_cpus, resolved_sht_threads)
    else:
        resolved_workers = workers
        resolved_sht_threads = (
            _ceil_division(available_cpus, workers)
            if sht_threads is None
            else sht_threads
        )
    return _ExecutionPolicy(
        workers=resolved_workers,
        sht_threads=resolved_sht_threads,
    )


def _available_cpu_count() -> int:
    """Return a positive process-appropriate CPU count when available."""
    process_cpu_count = getattr(os, "process_cpu_count", None)
    if process_cpu_count is not None:
        try:
            count = process_cpu_count()
        except (NotImplementedError, OSError):
            pass
        else:
            valid_count = _positive_cpu_count(count)
            if valid_count is not None:
                return valid_count

    sched_getaffinity = getattr(os, "sched_getaffinity", None)
    if sched_getaffinity is not None:
        try:
            count = len(sched_getaffinity(0))
        except (NotImplementedError, OSError):
            pass
        else:
            valid_count = _positive_cpu_count(count)
            if valid_count is not None:
                return valid_count

    valid_count = _positive_cpu_count(os.cpu_count())
    return 1 if valid_count is None else valid_count


def _positive_cpu_count(count: object) -> int | None:
    if isinstance(count, int) and count > 0:
        return count
    return None


def _ceil_division(dividend: int, divisor: int) -> int:
    return (dividend + divisor - 1) // divisor


def _validate_positive_option(value: int | None, name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer or None")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer or None")


@contextmanager
def _local_dask_executor(workers: int) -> Iterator[None]:
    """Own a temporary local Dask pool for one complete CLI operation."""
    try:
        import dask
    except ImportError as error:
        raise _CLIDependencyError(_DASK_CLI_INSTALL_MESSAGE) from error

    with (
        ThreadPoolExecutor(max_workers=workers) as executor,
        dask.config.set(pool=executor),
    ):
        yield


def _info(arguments: argparse.Namespace) -> int:
    with _open_dataset(arguments.input) as dataset:
        grid = detect_grid(dataset)
        print(f"grid_type: {grid.kind}")
        print(f"nlat: {grid.nlat}")
        print(f"nlon: {grid.nlon}")
    return 0


def _filter(
    arguments: argparse.Namespace,
    dataset: xr.Dataset,
    policy: _ExecutionPolicy,
) -> int:
    field = _select_variable(dataset, arguments.var)
    result = filter(
        field,
        arguments.truncation,
        lmin=arguments.lmin,
        lmax=arguments.lmax,
        taper=arguments.taper,
        sht_threads=policy.sht_threads,
    )
    _write_dataset(result.to_dataset(name=result.name or "filtered"), arguments.output)
    return 0


def _regrid(
    arguments: argparse.Namespace,
    dataset: xr.Dataset,
    policy: _ExecutionPolicy,
) -> int:
    field = _select_variable(dataset, arguments.var)
    constructor = gaussian_grid if arguments.grid == "gl" else clenshaw_curtis_grid
    target = constructor(
        arguments.nlat,
        arguments.nlon,
        lon0=arguments.lon0,
        latitude_order=arguments.latitude_order,
    )
    result = regrid(
        field,
        target,
        arguments.truncation,
        lmin=arguments.lmin,
        lmax=arguments.lmax,
        taper=arguments.taper,
        sht_threads=policy.sht_threads,
    )
    _write_dataset(result.to_dataset(name=result.name or "regridded"), arguments.output)
    return 0


def _kinematics(
    arguments: argparse.Namespace,
    dataset: xr.Dataset,
    policy: _ExecutionPolicy,
) -> int:
    result = kinematics(
        _wind_variable(dataset, "u", arguments.u),
        _wind_variable(dataset, "v", arguments.v),
        vorticity=arguments.vorticity,
        divergence=arguments.divergence,
        sht_threads=policy.sht_threads,
    )
    _write_dataset(result, arguments.output)
    return 0


def _potentials(
    arguments: argparse.Namespace,
    dataset: xr.Dataset,
    policy: _ExecutionPolicy,
) -> int:
    result = potentials(
        _wind_variable(dataset, "u", arguments.u),
        _wind_variable(dataset, "v", arguments.v),
        streamfunction=arguments.streamfunction,
        velocity_potential=arguments.velocity_potential,
        sht_threads=policy.sht_threads,
    )
    _write_dataset(result, arguments.output)
    return 0


def _wind(
    arguments: argparse.Namespace,
    dataset: xr.Dataset,
    policy: _ExecutionPolicy,
) -> int:
    if arguments.source == "vorticity_divergence":
        first = _wind_variable(dataset, "vo", arguments.vorticity)
        second = _wind_variable(dataset, "d", arguments.divergence)
    else:
        first = _wind_variable(dataset, "strf", arguments.streamfunction)
        second = _wind_variable(dataset, "vp", arguments.velocity_potential)
    result = wind(
        first,
        second,
        source=arguments.source,
        eastward=arguments.eastward,
        northward=arguments.northward,
        sht_threads=policy.sht_threads,
    )
    _write_dataset(result, arguments.output)
    return 0


def _is_zarr_path(path: str) -> bool:
    return Path(path).suffix.lower() == ".zarr"


def _is_grib_path(path: str) -> bool:
    return Path(path).suffix.lower() in {".grib", ".grib1", ".grib2", ".grb"}


def _grib_is_unsupported() -> bool:
    return (
        sys.platform == "win32"
        and sys.version_info.major == 3
        and sys.version_info.minor == 14
    )


def _open_dataset(path: str, *, chunks: str | None = None) -> xr.Dataset:
    """Open NetCDF, Zarr, or another xarray-supported input path."""
    if _is_zarr_path(path):
        if find_spec("zarr") is None:
            raise _CLIDependencyError(_CLI_INSTALL_MESSAGE)
        if chunks is not None:
            _require_dask()
        try:
            dataset = (
                xr.open_zarr(path)
                if chunks is None
                else xr.open_zarr(path, chunks=chunks)
            )
            assert isinstance(dataset, xr.Dataset)
            return dataset
        except ImportError as error:
            raise _CLIDependencyError(_CLI_INSTALL_MESSAGE) from error
    if _is_grib_path(path) and _grib_is_unsupported():
        raise _CLIDependencyError(_GRIB_WINDOWS_PY314_MESSAGE)
    if not _input_backend_available(path):
        raise _CLIDependencyError(_CLI_INSTALL_MESSAGE)
    if chunks is not None:
        _require_dask()
    try:
        return (
            xr.open_dataset(path)
            if chunks is None
            else xr.open_dataset(path, chunks=chunks)
        )
    except ImportError as error:
        raise _CLIDependencyError(_CLI_INSTALL_MESSAGE) from error
    except ValueError as error:
        if not _input_backend_available(path):
            raise _CLIDependencyError(_CLI_INSTALL_MESSAGE) from error
        raise


def _require_dask() -> None:
    try:
        import_module("dask")
    except ImportError as error:
        raise _CLIDependencyError(_DASK_CLI_INSTALL_MESSAGE) from error


def _write_dataset(dataset: xr.Dataset, output: str) -> None:
    """Write a Zarr store or NetCDF through the CLI h5netcdf backend."""
    if _is_zarr_path(output):
        if find_spec("zarr") is None:
            raise _CLIDependencyError(_CLI_INSTALL_MESSAGE)
        try:
            dataset.to_zarr(output, mode="w")
        except ImportError as error:
            raise _CLIDependencyError(_CLI_INSTALL_MESSAGE) from error
        return
    if find_spec("h5netcdf") is None:
        raise _CLIDependencyError(_CLI_INSTALL_MESSAGE)
    try:
        dataset.to_netcdf(output, engine="h5netcdf")
    except ImportError as error:
        raise _CLIDependencyError(_CLI_INSTALL_MESSAGE) from error
    except ValueError as error:
        if find_spec("h5netcdf") is None:
            raise _CLIDependencyError(_CLI_INSTALL_MESSAGE) from error
        raise


def _input_backend_available(path: str) -> bool:
    """Check the CLI backend expected for a failed generic input dispatch."""
    if _is_grib_path(path):
        return find_spec("cfgrib") is not None
    return find_spec("h5netcdf") is not None


def _package_version() -> str:
    """Return the installed package version without importing optional I/O."""
    try:
        return version("spharmgrid")
    except PackageNotFoundError:
        return "unknown"


def _select_variable(dataset: xr.Dataset, name: str | None) -> xr.DataArray:
    if name is not None:
        if name not in dataset.data_vars:
            raise ValueError(f"variable {name!r} is not a data variable")
        return dataset[name]
    variables = list(dataset.data_vars)
    if len(variables) != 1:
        raise ValueError(
            "--var is required when the input Dataset has multiple variables"
        )
    return dataset[variables[0]]


def _wind_variable(
    dataset: xr.Dataset, quantity: Quantity, explicit: str | None
) -> xr.DataArray:
    return find_variable(dataset, quantity, explicit)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
