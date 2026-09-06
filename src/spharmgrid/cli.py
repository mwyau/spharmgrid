"""Small file-oriented command line interface that delegates I/O to xarray."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
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
_GRIB_WINDOWS_PY314_MESSAGE = (
    "GRIB input is not supported on Windows with Python 3.14.\n"
    "Use Python 3.12 or 3.13 for GRIB input."
)


class _CLIDependencyError(RuntimeError):
    """Raised when a file operation needs the optional CLI dependencies."""


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``spharmgrid`` command line interface."""
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "info":
            return _info(arguments)
        if arguments.command == "filter":
            return _filter(arguments)
        if arguments.command == "regrid":
            return _regrid(arguments)
        if arguments.command == "kinematics":
            return _kinematics(arguments)
        if arguments.command == "potentials":
            return _potentials(arguments)
        if arguments.command == "wind":
            return _wind(arguments)
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
            "DUCC0-backed spherical-harmonic operations for xarray files. "
            "CLI I/O supports NetCDF and Zarr read/write plus GRIB input."
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
    filtered.add_argument(
        "--var", help="variable to filter; required for multi-variable input"
    )

    regridded = subparsers.add_parser("regrid", help="spectrally regrid one variable")
    _input_output_arguments(regridded)
    _spectral_arguments(regridded)
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
    kinematic.add_argument("--vorticity", default="vo", help="vorticity output name")
    kinematic.add_argument("--divergence", default="d", help="divergence output name")

    potential = subparsers.add_parser(
        "potentials", help="compute streamfunction and velocity potential"
    )
    _input_output_arguments(potential)
    _wind_input_arguments(potential)
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
    inverse.add_argument("--nthreads", type=int, help="DUCC threads per transform")
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
    parser.add_argument("--spectral", help="Tn or Tn-m retained range")
    parser.add_argument("--lmin", type=int, help="explicit lower retained degree")
    parser.add_argument("--lmax", type=int, help="explicit upper retained degree")
    parser.add_argument("--taper", type=float, help="response at upper retained degree")
    parser.add_argument("--nthreads", type=int, help="DUCC threads per transform")


def _wind_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--u", help="eastward-wind input variable")
    parser.add_argument("--v", help="northward-wind input variable")
    parser.add_argument("--nthreads", type=int, help="DUCC threads per transform")


def _info(arguments: argparse.Namespace) -> int:
    with _open_dataset(arguments.input) as dataset:
        grid = detect_grid(dataset)
        print(f"grid_type: {grid.kind}")
        print(f"nlat: {grid.nlat}")
        print(f"nlon: {grid.nlon}")
    return 0


def _filter(arguments: argparse.Namespace) -> int:
    with _open_dataset(arguments.input) as dataset:
        field = _select_variable(dataset, arguments.var)
        result = filter(
            field,
            arguments.spectral,
            lmin=arguments.lmin,
            lmax=arguments.lmax,
            taper=arguments.taper,
            nthreads=arguments.nthreads,
        )
        _write_dataset(
            result.to_dataset(name=result.name or "filtered"), arguments.output
        )
    return 0


def _regrid(arguments: argparse.Namespace) -> int:
    with _open_dataset(arguments.input) as dataset:
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
            arguments.spectral,
            lmin=arguments.lmin,
            lmax=arguments.lmax,
            taper=arguments.taper,
            nthreads=arguments.nthreads,
        )
        _write_dataset(
            result.to_dataset(name=result.name or "regridded"), arguments.output
        )
    return 0


def _kinematics(arguments: argparse.Namespace) -> int:
    with _open_dataset(arguments.input) as dataset:
        result = kinematics(
            _wind_variable(dataset, "u", arguments.u),
            _wind_variable(dataset, "v", arguments.v),
            vorticity=arguments.vorticity,
            divergence=arguments.divergence,
            nthreads=arguments.nthreads,
        )
        _write_dataset(result, arguments.output)
    return 0


def _potentials(arguments: argparse.Namespace) -> int:
    with _open_dataset(arguments.input) as dataset:
        result = potentials(
            _wind_variable(dataset, "u", arguments.u),
            _wind_variable(dataset, "v", arguments.v),
            streamfunction=arguments.streamfunction,
            velocity_potential=arguments.velocity_potential,
            nthreads=arguments.nthreads,
        )
        _write_dataset(result, arguments.output)
    return 0


def _wind(arguments: argparse.Namespace) -> int:
    with _open_dataset(arguments.input) as dataset:
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
            nthreads=arguments.nthreads,
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


def _open_dataset(path: str) -> xr.Dataset:
    """Open NetCDF, Zarr, or another xarray-supported input path."""
    if _is_zarr_path(path):
        if find_spec("zarr") is None:
            raise _CLIDependencyError(_CLI_INSTALL_MESSAGE)
        try:
            return xr.open_zarr(path)
        except ImportError as error:
            raise _CLIDependencyError(_CLI_INSTALL_MESSAGE) from error
    if _is_grib_path(path) and _grib_is_unsupported():
        raise _CLIDependencyError(_GRIB_WINDOWS_PY314_MESSAGE)
    if not _input_backend_available(path):
        raise _CLIDependencyError(_CLI_INSTALL_MESSAGE)
    try:
        return xr.open_dataset(path)
    except ImportError as error:
        raise _CLIDependencyError(_CLI_INSTALL_MESSAGE) from error
    except ValueError as error:
        if not _input_backend_available(path):
            raise _CLIDependencyError(_CLI_INSTALL_MESSAGE) from error
        raise


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
