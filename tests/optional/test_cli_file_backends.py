"""Command-line integration tests using optional xarray file backends."""

from __future__ import annotations

from pathlib import Path

import pytest
import xarray as xr

import spharmgrid as sg
from spharmgrid.cli import main
from tests.conftest import scalar_field, solid_body_wind, supported_grid

pytest.importorskip("h5netcdf")
pytest.importorskip("zarr")


def _write_netcdf(dataset: xr.Dataset, path: Path) -> None:
    dataset.to_netcdf(path, engine="h5netcdf")


def test_info_and_filter_commands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    grid = supported_grid("cc")
    field = scalar_field(grid, name="msl")
    input_path = tmp_path / "input.nc"
    output_path = tmp_path / "filtered.nc"
    _write_netcdf(field.to_dataset(), input_path)

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
                "--truncation",
                "T3",
            ]
        )
        == 0
    )
    with xr.open_dataset(output_path, engine="h5netcdf") as result:
        assert "msl" in result
        xr.testing.assert_allclose(result["msl"], sg.filter(field, "T3"))


def test_filter_reads_and_writes_zarr(tmp_path: Path) -> None:
    grid = supported_grid("cc")
    field = scalar_field(grid, name="msl")
    input_path = tmp_path / "input.zarr"
    output_path = tmp_path / "filtered.zarr"
    field.to_dataset().to_zarr(input_path, mode="w")

    assert (
        main(
            [
                "filter",
                str(input_path),
                str(output_path),
                "--var",
                "msl",
                "--truncation",
                "T3",
            ]
        )
        == 0
    )
    with xr.open_zarr(output_path) as result:
        assert "msl" in result
        xr.testing.assert_allclose(result["msl"], sg.filter(field, "T3"))


def test_kinematics_command_uses_dataset_variable_discovery(tmp_path: Path) -> None:
    grid = supported_grid("cc")
    u, v = solid_body_wind(grid)
    input_path = tmp_path / "wind.nc"
    output_path = tmp_path / "kinematics.nc"
    expected = sg.kinematics(u, v)
    _write_netcdf(xr.Dataset({"u": u, "v": v}), input_path)

    assert main(["kinematics", str(input_path), str(output_path)]) == 0
    with xr.open_dataset(output_path, engine="h5netcdf") as result:
        assert set(result.data_vars) == {"vo", "d"}
        for name in expected.data_vars:
            xr.testing.assert_allclose(result[name], expected[name])


def test_regrid_potentials_and_wind_commands(tmp_path: Path) -> None:
    grid = supported_grid("cc")
    scalar_input = tmp_path / "scalar.nc"
    regridded_output = tmp_path / "regridded.nc"
    field = scalar_field(grid, name="msl")
    target_grid = sg.gaussian_grid(12, 24)
    expected_regridded = sg.regrid(field, target_grid, truncation="T3")
    _write_netcdf(field.to_dataset(), scalar_input)

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
                "--truncation",
                "T3",
            ]
        )
        == 0
    )
    with xr.open_dataset(regridded_output, engine="h5netcdf") as result:
        assert result.sizes == {"lat": 12, "lon": 24}
        xr.testing.assert_allclose(result["msl"], expected_regridded)

    u, v = solid_body_wind(grid)
    wind_input = tmp_path / "wind.nc"
    potentials_output = tmp_path / "potentials.nc"
    diagnostics_input = tmp_path / "diagnostics.nc"
    reconstructed_output = tmp_path / "reconstructed.nc"
    expected_potentials = sg.potentials(u, v)
    _write_netcdf(xr.Dataset({"u": u, "v": v}), wind_input)
    assert main(["potentials", str(wind_input), str(potentials_output)]) == 0
    with xr.open_dataset(potentials_output, engine="h5netcdf") as result:
        assert set(result.data_vars) == {"strf", "vp"}
        for name in expected_potentials.data_vars:
            xr.testing.assert_allclose(result[name], expected_potentials[name])

    diagnostics = sg.kinematics(u, v)
    expected_wind = sg.wind(diagnostics.vo, diagnostics.d)
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
        for name in expected_wind.data_vars:
            xr.testing.assert_allclose(result[name], expected_wind[name])
