"""CLI delegation tests using xarray-supported file backends."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import xarray as xr

import spharmgrid as sg
from spharmgrid import cli
from spharmgrid.cli import main
from tests.conftest import scalar_field, solid_body_wind, supported_grid


def _write_netcdf(dataset: xr.Dataset, path: Path) -> None:
    dataset.to_netcdf(path, engine="h5netcdf")


def test_core_cli_help_and_version_do_not_need_file_backends(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as help_exit:
        main(["--help"])
    assert help_exit.value.code == 0
    assert "NetCDF and Zarr read/write plus GRIB input" in capsys.readouterr().out

    with pytest.raises(SystemExit) as version_exit:
        main(["--version"])
    assert version_exit.value.code == 0
    assert capsys.readouterr().out.startswith("spharmgrid ")


def test_import_does_not_load_optional_cli_backends() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import spharmgrid; "
                "assert not any(name in sys.modules for name in "
                "('h5netcdf', 'zarr', 'cfgrib'))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stderr == ""


def test_missing_cli_dependencies_are_actionable(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    def fail_open_dataset(path: str) -> xr.Dataset:
        raise ImportError(f"backend unavailable for {path}")

    monkeypatch.setattr(cli.xr, "open_dataset", fail_open_dataset)

    assert main(["info", str(tmp_path / "input.nc")]) == 2
    error = capsys.readouterr().err
    assert "CLI dependencies are not installed." in error
    assert 'pip install "spharmgrid[cli]"' in error
    assert "Traceback" not in error


def test_info_and_filter_commands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    grid = supported_grid("cc")
    input_path = tmp_path / "input.nc"
    output_path = tmp_path / "filtered.nc"
    _write_netcdf(scalar_field(grid, name="msl").to_dataset(), input_path)

    assert main(["info", str(input_path)]) == 0
    assert "grid_type: cc" in capsys.readouterr().out
    assert (
        main(
            [
                "filter",
                str(input_path),
                str(output_path),
                "--var",
                "msl",
                "--spectral",
                "T3",
            ]
        )
        == 0
    )
    with xr.open_dataset(output_path, engine="h5netcdf") as result:
        assert "msl" in result


def test_filter_reads_and_writes_zarr(tmp_path: Path) -> None:
    grid = supported_grid("cc")
    input_path = tmp_path / "input.zarr"
    output_path = tmp_path / "filtered.zarr"
    scalar_field(grid, name="msl").to_dataset().to_zarr(input_path, mode="w")

    assert (
        main(
            [
                "filter",
                str(input_path),
                str(output_path),
                "--var",
                "msl",
                "--spectral",
                "T3",
            ]
        )
        == 0
    )
    with xr.open_zarr(output_path) as result:
        assert "msl" in result


def test_kinematics_command_uses_dataset_variable_discovery(tmp_path: Path) -> None:
    grid = supported_grid("cc")
    u, v = solid_body_wind(grid)
    input_path = tmp_path / "wind.nc"
    output_path = tmp_path / "kinematics.nc"
    _write_netcdf(xr.Dataset({"u": u, "v": v}), input_path)

    assert main(["kinematics", str(input_path), str(output_path)]) == 0
    with xr.open_dataset(output_path, engine="h5netcdf") as result:
        assert set(result.data_vars) == {"vo", "d"}


def test_regrid_potentials_and_wind_commands(tmp_path: Path) -> None:
    grid = supported_grid("cc")
    scalar_input = tmp_path / "scalar.nc"
    regridded_output = tmp_path / "regridded.nc"
    _write_netcdf(scalar_field(grid, name="msl").to_dataset(), scalar_input)

    assert (
        main(
            [
                "regrid",
                str(scalar_input),
                str(regridded_output),
                "--var",
                "msl",
                "--grid",
                "gl",
                "--nlat",
                "12",
                "--nlon",
                "24",
                "--spectral",
                "T3",
            ]
        )
        == 0
    )
    with xr.open_dataset(regridded_output, engine="h5netcdf") as result:
        assert result.sizes == {"lat": 12, "lon": 24}

    u, v = solid_body_wind(grid)
    wind_input = tmp_path / "wind.nc"
    potentials_output = tmp_path / "potentials.nc"
    diagnostics_input = tmp_path / "diagnostics.nc"
    reconstructed_output = tmp_path / "reconstructed.nc"
    _write_netcdf(xr.Dataset({"u": u, "v": v}), wind_input)
    assert main(["potentials", str(wind_input), str(potentials_output)]) == 0
    with xr.open_dataset(potentials_output, engine="h5netcdf") as result:
        assert set(result.data_vars) == {"strf", "vp"}

    diagnostics = sg.kinematics(u, v)
    _write_netcdf(diagnostics, diagnostics_input)
    assert (
        main(
            [
                "wind",
                str(diagnostics_input),
                str(reconstructed_output),
                "--source",
                "vorticity_divergence",
            ]
        )
        == 0
    )
    with xr.open_dataset(reconstructed_output, engine="h5netcdf") as result:
        assert set(result.data_vars) == {"u", "v"}
